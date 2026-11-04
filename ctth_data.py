"""
ctth_data.py — Outils communs du projet CTTH (altitude du sommet des nuages, MTG-FCI x EarthCARE).

Regroupe ce qui sert dans tous les notebooks, pour ne plus le recopier :
    - chargement de data_cloud_top_V2.pkl avec les noms explicites du dictionnaire des variables ;
    - cube des fenêtres brutes bt_win11 (N, 6, 11, 11) ;
    - extraction des vignettes 5x5 brutes ou corrigées de la parallaxe ;
    - construction des variables (statistiques, contrastes physiques, texture, ondelettes, inversion ERA5) ;
    - garde-fou contre les variables dérivées de la cible ;
    - métriques correctes (vrai RMSE, vrai R2) ;
    - validation croisée par jour et comparaison appariée jour par jour ;
    - histogramme 2D prédiction / EarthCARE.

Conventions (vérifiées sur les données) :
    - fenêtre 11x11 : axe 0 = longitude (indice i), axe 1 = latitude (indice j), centre [5, 5] = pixel au sol ;
    - pixel corrigé de la parallaxe : [5 + parallax_shift_lon_px, 5 + parallax_shift_lat_px] ;
    - vignette V1 (25 valeurs) = transposée de la vignette 5x5 corrigée.

Usage minimal :
    from ctth_data import load_v2, build_features, FAMILIES_V2, cross_validate_by_day
    df, win = load_v2('./Data/data_cloud_top_V2.pkl')
    df = build_features(df, win)
    res = cross_validate_by_day(df, FAMILIES_V2['E'], make_model=my_factory)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------------------------

CH_ORIG = ['WV_063', 'WV_073', 'IR_087', 'IR_105', 'IR_123', 'IR_133']
CH = ['wv063', 'wv073', 'ir087', 'ir105', 'ir123', 'ir133']
P_LEVELS = [1000, 850, 700, 500, 300, 200, 100]
# Altitudes approximatives des niveaux de pression (atmosphère standard, m) — pas de géopotentiel dans les données
Z_LEVELS_STD = np.array([110, 1460, 3010, 5570, 9160, 11780, 16180], dtype=float)
SAF_FILL = 63535

TARGET = 'cth_earthcare_m'
SAF = 'cth_safnwc_m'
CT = 'cloud_type_safnwc'
DATE = 'date_utc'

CT_NAMES = {5: 'Very low', 6: 'Low', 7: 'Mid-level', 8: 'High', 9: 'Very high', 10: 'Fractional',
            11: 'Semitransp. thin', 12: 'Semitransp. mod. thick', 13: 'Semitransp. thick',
            14: 'Semitransp. above low/mid', 15: 'Semitransp. above snow'}

ALT_BINS = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 17000]

# Toute variable dont le nom contient l'un de ces motifs est dérivée de la cible : interdite en opérationnel
FORBIDDEN_PATTERNS = ('p5corr', 'parallax_shift', 'earthcare')


# ----------------------------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------------------------

def _rename_map() -> dict:
    m = {}
    for o, c in zip(CH_ORIG, CH):
        m[o + '_ClearSky'] = f'bt_{c}_clearsky'
        m[o + '_C5'] = f'bt_{c}_min_p5corr'
        m[o + '_W5'] = f'bt_{c}_max_p5corr'
        m[o + '_std5'] = f'bt_{c}_std_p5corr'
    m |= {f'T_{p}': f't_era5_{p}hpa' for p in P_LEVELS}
    m |= {'nb_pix_LON_cor_parallax': 'parallax_shift_lon_px', 'nb_pix_LAT_cor_parallax': 'parallax_shift_lat_px',
          'Cloud_Top_Altitude_EarthCARE': TARGET, 'Cloud_Top_Altitude_SAFNWC': SAF,
          'Cloud_Type_SAFNWC': CT, 'Day_UTC': DATE, 'HH_UTC': 'hour_utc', 'MN_UTC': 'minute_utc'}
    return m


def load_v2(path: str, dtype: str = 'float32') -> tuple[pd.DataFrame, np.ndarray]:
    """Charge data_cloud_top_V2.pkl.

    Retourne
    --------
    df  : DataFrame des colonnes scalaires, renommées selon le dictionnaire, types explicites,
          altitude SAFNWC = NaN là où elle vaut 63535 (type 10).
    win : cube des fenêtres brutes, forme (N, 6, 11, 11), canaux dans l'ordre de CH.
    """
    raw = pd.read_pickle(path).reset_index(drop=True)
    missing = [c for c in CH_ORIG if c not in raw.columns]
    if missing:
        raise ValueError(f'Colonnes de fenêtres absentes : {missing} (fichier V1 au lieu de V2 ?)')
    win = np.stack([np.stack(raw[o].values) for o in CH_ORIG], axis=1).astype(dtype)
    if win.shape[1:] != (6, 11, 11):
        raise ValueError(f'Fenêtres de forme inattendue : {win.shape}')

    df = raw.drop(columns=CH_ORIG).rename(columns=_rename_map())
    df[DATE] = df[DATE].astype(int)
    for c in CH:
        df[f'bt_{c}_clearsky'] = df[f'bt_{c}_clearsky'].astype(float)
    df[SAF] = df[SAF].astype(float).where(df[SAF] != SAF_FILL)
    df['alt_bin'] = pd.cut(df[TARGET], ALT_BINS)
    return df, win


def channel(name: str) -> int:
    """Indice d'un canal dans le cube : channel('ir105') -> 3."""
    return CH.index(name)


# ----------------------------------------------------------------------------------------------
# Vignettes
# ----------------------------------------------------------------------------------------------

def patch_raw(win: np.ndarray, size: int = 5) -> np.ndarray:
    """Vignette brute size x size centrée sur le pixel au sol. Forme (N, 6, size, size)."""
    h = size // 2
    return win[:, :, 5 - h:6 + h, 5 - h:6 + h]


def patch_corrected(win: np.ndarray, df: pd.DataFrame, size: int = 5) -> np.ndarray:
    """Vignette size x size centrée sur le pixel corrigé de la parallaxe (DÉRIVÉE DE LA CIBLE).
    À réserver au scénario « alignement parfait ». Forme (N, 6, size, size)."""
    h = size // 2
    i0 = 5 + df['parallax_shift_lon_px'].to_numpy()
    j0 = 5 + df['parallax_shift_lat_px'].to_numpy()
    if (i0 - h).min() < 0 or (i0 + h).max() > 10 or (j0 - h).min() < 0 or (j0 + h).max() > 10:
        raise ValueError(f'Une vignette {size}x{size} corrigée sort de la fenêtre 11x11')
    n = len(win); k = np.arange(n)
    out = np.empty((n, win.shape[1], size, size), dtype=win.dtype)
    for a, di in enumerate(range(-h, h + 1)):
        for b, dj in enumerate(range(-h, h + 1)):
            out[:, :, a, b] = win[k, :, i0 + di, j0 + dj]
    return out


def patch_v1_order(p5: np.ndarray) -> np.ndarray:
    """Convertit une vignette (N, 5, 5) de la V2 dans l'ordre aplati de la V1 (N, 25) : transposition."""
    return p5.transpose(0, 2, 1).reshape(len(p5), 25)


def detrend_plane(w: np.ndarray) -> np.ndarray:
    """Retire un plan ajusté par moindres carrés à chaque fenêtre (N, H, W).
    À appliquer avant toute analyse de texture : le champ est dominé par un gradient à grande échelle."""
    n, H, W = w.shape
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    A = np.column_stack([np.ones(H * W), ii.ravel(), jj.ravel()])          # (HW, 3)
    P = np.linalg.pinv(A)                                                   # (3, HW)
    flat = w.reshape(n, -1)
    coef = flat @ P.T                                                       # (n, 3)
    return (flat - coef @ A.T).reshape(n, H, W)


# ----------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------

def invert_era5_profile(bt: np.ndarray, T: np.ndarray, z: np.ndarray = Z_LEVELS_STD) -> np.ndarray:
    """Altitude (m) où le profil de température atteint bt, en montant depuis la surface.
    bt : (N,), T : (N, 7) températures aux niveaux P_LEVELS. Plus chaud que la surface -> z[0] ;
    plus froid que tout le profil -> altitude du minimum (tropopause)."""
    out = np.full(len(bt), np.nan)
    done = bt >= T[:, 0]
    out[done] = z[0]
    for lv in range(len(z) - 1):
        m = ~done & (T[:, lv] >= bt) & (bt > T[:, lv + 1])
        out[m] = z[lv] + (T[m, lv] - bt[m]) / (T[m, lv] - T[m, lv + 1]) * (z[lv + 1] - z[lv])
        done |= m
    out[~done] = z[np.argmin(T[~done], axis=1)]
    return out


def texture_features(w: np.ndarray, prefix: str) -> pd.DataFrame:
    """Gradient moyen et cohérence du tenseur de structure (invariante à l'orientation) sur une fenêtre (N, H, W)."""
    gi, gj = np.gradient(w, axis=(1, 2))
    Jii, Jjj, Jij = (gi * gi).mean((1, 2)), (gj * gj).mean((1, 2)), (gi * gj).mean((1, 2))
    return pd.DataFrame({f'grad_{prefix}': np.hypot(gi, gj).mean((1, 2)),
                         f'coherence_{prefix}': np.sqrt((Jii - Jjj) ** 2 + 4 * Jij ** 2) / (Jii + Jjj + 1e-9)})


def wavelet_energy(w: np.ndarray, prefix: str, wavelet: str = 'haar', level: int = 2) -> pd.DataFrame:
    """Énergie des coefficients d'une transformée en ondelettes stationnaire, par niveau et direction.
    Fenêtre 11x11 complétée à 12x12 par réflexion. Nécessite PyWavelets."""
    import pywt
    a = np.pad(w, ((0, 0), (0, 1), (0, 1)), mode='reflect')
    a = a - a.mean((1, 2), keepdims=True)
    coeffs = pywt.swt2(a, wavelet, level=level, axes=(1, 2), trim_approx=True)
    out = {}
    for lev, (cH, cV, cD) in zip(range(level, 0, -1), coeffs[1:]):
        out[f'wav_{prefix}_l{lev}_h'] = (cH ** 2).mean((1, 2))
        out[f'wav_{prefix}_l{lev}_v'] = (cV ** 2).mean((1, 2))
        out[f'wav_{prefix}_l{lev}_d'] = (cD ** 2).mean((1, 2))
    return pd.DataFrame(out)


def build_features(df: pd.DataFrame, win: np.ndarray, wavelets: bool = True) -> pd.DataFrame:
    """Ajoute à df toutes les variables de l'EDA V2 (scénario opérationnel, sauf colonnes _p5corr d'origine).

    - bt_<c>_center                         pixel au sol
    - bt_<c>_center_p5corr                  pixel corrigé de la parallaxe (FUITE : référence seulement)
    - bt_<c>_{min,max,std}_p5raw            vignette brute 5x5
    - bt_<c>_{min,max,std,q10}_win11        fenêtre brute 11x11
    - dbt_<c>_cs                            centre - ciel clair
    - dbt_ir105_ir123, dbt_wv063_ir105, dbt_ir133_ir105
    - cth_era5inv_m                         inversion du profil ERA5
    - grad_ir105_win11, coherence_ir105_win11
    - wav_ir105_l{1,2}_{h,v,d}              (si wavelets=True)
    """
    df = df.copy()
    n = len(df)
    new = {}
    for ci, c in enumerate(CH):
        w = win[:, ci]
        p5 = w[:, 3:8, 3:8].reshape(n, 25)
        p11 = w.reshape(n, 121)
        new[f'bt_{c}_center'] = w[:, 5, 5]
        new[f'bt_{c}_min_p5raw'], new[f'bt_{c}_max_p5raw'], new[f'bt_{c}_std_p5raw'] = p5.min(1), p5.max(1), p5.std(1)
        new[f'bt_{c}_min_win11'], new[f'bt_{c}_max_win11'] = p11.min(1), p11.max(1)
        new[f'bt_{c}_std_win11'], new[f'bt_{c}_q10_win11'] = p11.std(1), np.quantile(p11, .1, axis=1)
        new[f'dbt_{c}_cs'] = w[:, 5, 5] - df[f'bt_{c}_clearsky'].to_numpy()
    new['dbt_ir105_ir123'] = new['bt_ir105_center'] - new['bt_ir123_center']
    new['dbt_wv063_ir105'] = new['bt_wv063_center'] - new['bt_ir105_center']
    new['dbt_ir133_ir105'] = new['bt_ir133_center'] - new['bt_ir105_center']
    # Pixel central corrigé de la parallaxe (DÉRIVÉ DE LA CIBLE : scénario « alignement parfait » seulement)
    k = np.arange(n)
    i0 = 5 + df['parallax_shift_lon_px'].to_numpy(); j0 = 5 + df['parallax_shift_lat_px'].to_numpy()
    for ci, c in enumerate(CH):
        new[f'bt_{c}_center_p5corr'] = win[k, ci, i0, j0]
    T = df[[f't_era5_{p}hpa' for p in P_LEVELS]].to_numpy(float)
    new['cth_era5inv_m'] = invert_era5_profile(new['bt_ir105_center'].astype(float), T)
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    w105 = win[:, channel('ir105')].astype(float)
    df = pd.concat([df, texture_features(w105, 'ir105_win11').set_index(df.index)], axis=1)
    if wavelets:
        df = pd.concat([df, wavelet_energy(w105, 'ir105').set_index(df.index)], axis=1)
    return df


# ----------------------------------------------------------------------------------------------
# Descripteurs spatiaux invariants à l'orientation : Gabor et scattering simplifié
# ----------------------------------------------------------------------------------------------
# Calculés sur la fenêtre 11x11 brute après retrait de la pente (detrend_plane), pour chaque canal.
# Filtres de Morlet (Gabor de moyenne nulle) à 3 longueurs d'onde x 8 orientations ; seules des
# grandeurs invariantes à la rotation sont gardées, car le sens du nord dans la fenêtre n'est pas
# confirmé et l'orientation dominante observée dans l'EDA est un artefact de grille.

GABOR_LAMBDAS = (3, 4, 6)          # longueurs d'onde en pixels (échelles utiles d'après le spectre de l'EDA)
GABOR_N_THETA = 8
_CORE = slice(2, 9)                # zone centrale 7x7 : limite les effets de bord
_FFT_SIZE = 21                     # 11 + 11 - 1 : convolution linéaire sans repliement


def morlet_kernel(lam: float, theta: float, size: int = 11, sigma_ratio: float = 0.56) -> np.ndarray:
    """Filtre de Morlet 2D (Gabor complexe de moyenne nulle), normalisé en énergie."""
    s = sigma_ratio * lam
    h = size // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(float)
    env = np.exp(-(xx ** 2 + yy ** 2) / (2 * s ** 2))
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    g = env * np.exp(1j * 2 * np.pi * xr / lam)
    g = g - env * (g.sum() / env.sum())
    return g / np.sqrt((np.abs(g) ** 2).sum())


def gabor_bank(lambdas=GABOR_LAMBDAS, n_theta: int = GABOR_N_THETA) -> np.ndarray:
    """Banque de filtres, forme (len(lambdas), n_theta, 11, 11)."""
    th = np.arange(n_theta) * np.pi / n_theta
    return np.stack([morlet_kernel(l, t) for l in lambdas for t in th]).reshape(len(lambdas), n_theta, 11, 11)


def _abs_conv(a: np.ndarray, Kf: np.ndarray) -> np.ndarray:
    """|a * K| (convolution 'same') pour a (n, 11, 11) et la FFT des filtres Kf (k, 21, 21) -> (n, k, 11, 11)."""
    import scipy.fft as sfft
    X = sfft.fft2(a, s=(_FFT_SIZE, _FFT_SIZE), axes=(-2, -1), workers=-1)
    R = sfft.ifft2(X[:, None] * Kf[None], axes=(-2, -1), workers=-1)
    return np.abs(R[..., 5:16, 5:16])


def spatial_descriptors(w: np.ndarray, prefix: str, batch: int = 2000) -> dict:
    """Descripteurs spatiaux d'un canal. w : (n, 11, 11), pente déjà retirée.

    gabor_<prefix>_l<λ>_e       énergie moyenne sur les 8 orientations, zone centrale 7x7
    gabor_<prefix>_l<λ>_aniso   (max - min) / somme des énergies sur les orientations : structure orientée ou non
    scat_<prefix>_l<λ1>x<λ2>_par  scattering d'ordre 2 || w * ψ(λ1, θ) | * ψ(λ2, θ) |, même orientation
    scat_<prefix>_l<λ1>x<λ2>_all  idem, moyenne sur toutes les paires d'orientations
    """
    import scipy.fft as sfft
    K = gabor_bank()
    L, T = K.shape[:2]
    Kf_all = sfft.fft2(K.reshape(L * T, 11, 11), s=(_FFT_SIZE, _FFT_SIZE), axes=(-2, -1))
    Kf_lev = [sfft.fft2(K[a], s=(_FFT_SIZE, _FFT_SIZE), axes=(-2, -1)) for a in range(L)]
    pairs = [(a1, a2) for a1 in range(L) for a2 in range(L) if GABOR_LAMBDAS[a2] > GABOR_LAMBDAS[a1]]
    n = len(w)
    res_g = np.zeros((n, L, 2)); res_s = np.zeros((n, len(pairs), 2)); diag = np.arange(T)
    for s0 in range(0, n, batch):
        sl = slice(s0, min(n, s0 + batch)); a = w[sl]; m = len(a)
        U = _abs_conv(a, Kf_all).reshape(m, L, T, 11, 11)
        E = (U[..., _CORE, _CORE] ** 2).mean((-2, -1))
        res_g[sl, :, 0] = E.mean(2)
        res_g[sl, :, 1] = (E.max(2) - E.min(2)) / (E.sum(2) + 1e-12)
        for p, (a1, a2) in enumerate(pairs):
            V = _abs_conv(U[:, a1].reshape(m * T, 11, 11), Kf_lev[a2])[..., _CORE, _CORE].mean((-2, -1)).reshape(m, T, T)
            res_s[sl, p, 0] = V[:, diag, diag].mean(1)
            res_s[sl, p, 1] = V.mean((1, 2))
    out = {}
    for i, l in enumerate(GABOR_LAMBDAS):
        out[f'gabor_{prefix}_l{l}_e'] = res_g[:, i, 0]
        out[f'gabor_{prefix}_l{l}_aniso'] = res_g[:, i, 1]
    for p, (a1, a2) in enumerate(pairs):
        tag = f'l{GABOR_LAMBDAS[a1]}x{GABOR_LAMBDAS[a2]}'
        out[f'scat_{prefix}_{tag}_par'] = res_s[:, p, 0]
        out[f'scat_{prefix}_{tag}_all'] = res_s[:, p, 1]
    return out


def spatial_features(win: np.ndarray, cache_path: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Gabor + scattering pour les 6 canaux (72 variables). Long : ~15-20 min sur 2 cœurs.
    Si cache_path est donné et existe, le recharge ; sinon calcule puis l'enregistre (format parquet)."""
    import os, time
    if cache_path and os.path.exists(cache_path):
        feat = pd.read_parquet(cache_path)
        if len(feat) != len(win):
            raise ValueError(f'Cache {cache_path} : {len(feat)} lignes au lieu de {len(win)}')
        return feat
    out = {}
    for ci, c in enumerate(CH):
        t = time.time()
        out |= spatial_descriptors(detrend_plane(win[:, ci].astype(float)), c)
        if verbose:
            print(f'{c} : {time.time() - t:.0f} s')
    feat = pd.DataFrame(out).astype('float32')
    if cache_path:
        feat.to_parquet(cache_path)
    return feat


GABOR_FEATURES = [f'gabor_{c}_l{l}_{k}' for c in CH for l in GABOR_LAMBDAS for k in ('e', 'aniso')]
SCAT_FEATURES = [f'scat_{c}_l{a}x{b}_{k}' for c in CH for a, b in ((3, 4), (3, 6), (4, 6)) for k in ('par', 'all')]


# Familles de variables de l'EDA V2 (section 7)
_BASE = [f'bt_{c}_center' for c in CH] + [f'bt_{c}_clearsky' for c in CH] + [f't_era5_{p}hpa' for p in P_LEVELS]
_P5CORR = [f'bt_{c}_{s}_p5corr' for c in CH for s in ['min', 'max', 'std']]
_P5RAW = [f'bt_{c}_{s}_p5raw' for c in CH for s in ['min', 'max', 'std']]
_W11 = [f'bt_{c}_{s}_win11' for c in CH for s in ['min', 'max', 'std', 'q10']]
_PHYS = (['dbt_ir105_ir123', 'dbt_wv063_ir105', 'dbt_ir133_ir105'] + [f'dbt_{c}_cs' for c in CH]
         + ['cth_era5inv_m', 'grad_ir105_win11', 'coherence_ir105_win11']
         + [f'wav_ir105_l{l}_{d}' for l in (2, 1) for d in 'hvd'])

FAMILIES_V2 = {
    'A': _BASE,                              # centre + ciel clair + ERA5
    'B': _BASE + _P5CORR,                    # + stats 5x5 corrigée (FUITE, référence seulement)
    'C': _BASE + _P5RAW,                     # + stats 5x5 brute
    'D': _BASE + _P5RAW + _W11,              # + stats 11x11 brute
    'E': _BASE + _P5RAW + _W11 + _PHYS,      # + physique et texture (base de comparaison : 736 m)
}
# Descripteurs spatiaux (nécessitent spatial_features)
FAMILIES_V2['F'] = FAMILIES_V2['E'] + GABOR_FEATURES                  # E + Gabor
FAMILIES_V2['G'] = FAMILIES_V2['E'] + SCAT_FEATURES                   # E + scattering
FAMILIES_V2['H'] = FAMILIES_V2['E'] + GABOR_FEATURES + SCAT_FEATURES  # E + les deux

# Les 37 variables du script de référence ctth_IA.py, avec les noms explicites (vignette corrigée = fuite)
REFERENCE_FEATURES = ([f'bt_{c}_center_p5corr' for c in CH] + [f'bt_{c}_clearsky' for c in CH]
                      + [f'bt_{c}_{s}_p5corr' for c in CH for s in ['min', 'max', 'std']]
                      + [f't_era5_{p}hpa' for p in P_LEVELS])


def check_features(features: list[str], operational: bool = True) -> list[str]:
    """Vérifie une liste de variables. En scénario opérationnel, lève une erreur si une variable
    dérivée de la cible (p5corr, parallax_shift, earthcare) ou la référence SAFNWC est présente."""
    bad = [f for f in features if any(p in f for p in FORBIDDEN_PATTERNS)]
    if TARGET in features:
        raise ValueError('La cible fait partie des variables d\'entrée')
    if operational and bad:
        raise ValueError(f'Variables interdites en scénario opérationnel : {bad}')
    if len(set(features)) != len(features):
        raise ValueError('Variables en double dans la liste')
    return bad


# ----------------------------------------------------------------------------------------------
# Métriques
# ----------------------------------------------------------------------------------------------

def metrics(y_true, y_pred) -> pd.Series:
    """MAE, biais (prédit - vrai), vrai RMSE et vrai R2 (coefficient de détermination), en ignorant les NaN."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    e = y_pred[m] - y_true[m]
    yt = y_true[m]
    return pd.Series({'N': int(m.sum()), 'MAE': np.abs(e).mean(), 'Biais': e.mean(),
                      'RMSE': np.sqrt((e ** 2).mean()),
                      'R2': 1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum()})


def metrics_by(df: pd.DataFrame, y_pred, by: str = CT) -> pd.DataFrame:
    """Métriques par groupe (type de nuage par défaut, ou 'alt_bin')."""
    tmp = pd.DataFrame({'y': df[TARGET].to_numpy(), 'p': np.asarray(y_pred, float), 'g': df[by].to_numpy()})
    return tmp.groupby('g', observed=True).apply(lambda t: metrics(t['y'], t['p']), include_groups=False)


# ----------------------------------------------------------------------------------------------
# Validation croisée par jour
# ----------------------------------------------------------------------------------------------

def cross_validate_by_day(df: pd.DataFrame, features: list[str], make_model, operational: bool = True,
                          target: str = TARGET, verbose: bool = False) -> dict:
    """Validation croisée « un jour = un pli » : chaque jour est prédit par un modèle entraîné sur les autres.

    make_model : fonction sans argument qui renvoie un modèle neuf ayant .fit(X, y) et .predict(X)
                 (pas d'arrêt précoce sur le jour de test).
    Retourne un dict : 'pred' (prédictions hors pli, alignées sur df), 'by_day' (MAE par jour),
                       'summary' (métriques globales + écart-type entre jours).
    """
    check_features(features, operational=operational)
    X = df[features].to_numpy(np.float32)
    y = df[target].to_numpy(float)
    days = df[DATE].to_numpy()
    pred = np.full(len(df), np.nan)
    for d in np.unique(days):
        te = days == d
        model = make_model()
        model.fit(X[~te], y[~te])
        pred[te] = np.ravel(model.predict(X[te]))
        if verbose:
            print(f'{d} : MAE {np.abs(pred[te] - y[te]).mean():.0f} m')
    by_day = pd.Series(np.abs(pred - y)).groupby(days).mean()
    summary = metrics(y, pred)
    summary['MAE écart-type entre jours'] = by_day.std()
    return {'pred': pred, 'by_day': by_day, 'summary': summary}


def paired_comparison(by_day_a: pd.Series, by_day_b: pd.Series) -> pd.Series:
    """Compare deux modèles jour par jour (mêmes plis). Gain = MAE(a) - MAE(b) : positif si b est meilleur.
    p_sign = probabilité, si b n'apportait rien, d'avoir au moins autant de jours gagnants (test du signe unilatéral)."""
    from math import comb
    diff = (by_day_a - by_day_b).dropna()
    n, wins = len(diff), int((diff > 0).sum())
    p = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n
    return pd.Series({'gain moyen (m)': diff.mean(), 'écart-type (m)': diff.std(),
                      'jours gagnants': f'{wins} / {n}', 'p_sign': p})


# ----------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------

def plot_hist2d(y_true, y_pred, title: str = '', ylabel: str = 'Prédit (m)', step: int = 250,
                vmax: float | None = None, ax=None, path: str | None = None):
    """Histogramme 2D prédiction / EarthCARE avec les vraies métriques (remplace les 6 blocs de ctth_IA.py)."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    bins = np.arange(0, 20000 + step, step)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    h = ax.hist2d(y_true[m], y_pred[m], bins=bins, cmap='viridis', norm=mpl.colors.LogNorm(vmax=vmax), cmin=1)
    plt.colorbar(h[3], ax=ax, label='Pixels')
    ax.plot([0, 20000], [0, 20000], 'k--', lw=1)
    ax.set_xlim(0, 20000); ax.set_ylim(0, 20000)
    s = metrics(y_true, y_pred)
    ax.text(800, 19000, f"N : {s['N']}\nMAE : {s['MAE']:.0f} m\nBiais : {s['Biais']:.0f} m\n"
                        f"RMSE : {s['RMSE']:.0f} m\nR² : {s['R2']:.3f}", va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=.85))
    ax.set_xlabel('EarthCARE-CPR (m)'); ax.set_ylabel(ylabel); ax.set_title(title)
    if own:
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=200)
    return ax