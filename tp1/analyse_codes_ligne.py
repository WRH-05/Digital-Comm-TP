# -*- coding: utf-8 -*-
"""
TP 1 - CODES EN LIGNE
Communications Numeriques - 4eme GEI, ELN, USTHB

ATTENTION : Les 4 signaux generes sont MELANGES.
Vous devez d'abord IDENTIFIER quel signal correspond a quel code
avant de repondre aux questions du TP.

Codes a identifier : NRZ Unipolaire, NRZ Bipolaire,
                     RZ Unipolaire (50%), Manchester

@author: Mr. MEFTAH
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

print("=" * 60)
print("              TP CODES EN LIGNE")
print("=" * 60)
print()
print("  ATTENTION : Les signaux sont MELANGES !")
print("  Identifiez chaque signal avant de repondre")
print("  aux questions du TP.")
print()

# ============================================================
#                    PARAMETRES
# ============================================================

Rb = 1000
Fs = 10000
N_bits = 20
N_bits_dsp = 10000

Tb = 1 / Rb
sps = int(Fs / Rb)
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1])
t = np.arange(0, N_bits * sps) / Fs * 1000

np.random.seed(42)
bits_dsp = np.random.randint(0, 2, N_bits_dsp)

# ============================================================
#           COULEURS
# ============================================================

couleur1 = '#E74C3C'   # Rouge
couleur2 = '#2ECC71'   # Vert
couleur3 = '#3498DB'   # Bleu
couleur4 = '#9B59B6'   # Violet

# ============================================================
#           FONCTION DE GENERATION (NE PAS MODIFIER)
# ============================================================

def generer_signaux(sequence, sps):
    N = len(sequence)
    s1 = np.zeros(N * sps)
    s2 = np.zeros(N * sps)
    s3 = np.zeros(N * sps)
    s4 = np.zeros(N * sps)

    for i in range(N):
        start = i * sps
        end = (i + 1) * sps
        half = start + sps // 2

        s1[start:end] = 2 * sequence[i] - 1

        if sequence[i] == 1:
            s2[start:half] = 1
            s2[half:end] = -1
        else:
            s2[start:half] = -1
            s2[half:end] = 1

        s3[start:end] = sequence[i]

        if sequence[i] == 1:
            s4[start:half] = 1

    return s1, s2, s3, s4

sig_1, sig_2, sig_3, sig_4 = generer_signaux(bits, sps)
sig_1_dsp, sig_2_dsp, sig_3_dsp, sig_4_dsp = generer_signaux(bits_dsp, sps)

# ============================================================
#  MANIPULATION 1 : Parametres fondamentaux (Question 1.1, 1.2)
# ============================================================

print("=" * 60)
print("  MANIPULATION 1 : Parametres fondamentaux")
print("=" * 60)
print()
print("  Determinez les parametres suivants a partir du code")
print("  et des figures :")
print()
print("  1.1 Parametres fondamentaux :")
print(f"      Debit binaire (Rb)              = __________ bit/s")
print(f"      Duree d'un bit (Tb)             = __________ ms")
print(f"      Frequence d'echantillonnage (Fs) = __________ Hz")
print(f"      Nombre de bits                   = __________")
print()
print("  1.2 Analyse technique :")
print(f"      Rapport Fs/Rb                    = __________ (oversampling factor)")
print(f"      Duree totale de la sequence      = __________ ms")
print(f"      Sequence binaire (20 bits)       = __________")
print()

# ============================================================
#  MANIPULATION 2 : Signaux temporels (Questions 2.1 a 2.4)
# ============================================================

print("=" * 60)
print("  MANIPULATION 2 : Analyse des Signaux Temporels")
print("=" * 60)
print()

# --- Figure 01_signaux_temporels.png ---

fig1, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

axes[0].step(range(N_bits), bits, where='post', color='black', linewidth=2)
axes[0].set_ylabel('Bits')
axes[0].set_title('Sequence binaire de reference', fontweight='bold', fontsize=12)
axes[0].set_ylim(-0.3, 1.5)
axes[0].set_yticks([0, 1])
axes[0].grid(True)
for i in range(N_bits):
    axes[0].text(i + 0.5, 1.2, str(bits[i]), ha='center',
                 fontsize=9, fontweight='bold')

sigs_plot = [sig_1, sig_2, sig_3, sig_4]
couleurs_plot = [couleur1, couleur2, couleur3, couleur4]
titres_plot = ['Signal 1 - Code : ?', 'Signal 2 - Code : ?',
               'Signal 3 - Code : ?', 'Signal 4 - Code : ?']

for k in range(4):
    ax = axes[k + 1]
    ax.plot(t, sigs_plot[k], color=couleurs_plot[k], linewidth=2)
    ax.set_ylabel('Amplitude (V)')
    ax.set_title(titres_plot[k], fontweight='bold', fontsize=11)
    ax.set_ylim(-1.8, 1.8)
    ax.grid(True)
    ax.axhline(y=0, color=[0.5, 0.5, 0.5], linestyle='--')
    for b in range(N_bits + 1):
        ax.axvline(x=b * Tb * 1000, color=[0.7, 0.7, 0.7], linestyle=':')

axes[4].set_xlabel('Temps (ms)')

fig1.suptitle('SIGNAUX TEMPORELS - Identifiez chaque code',
              fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('01_signaux_temporels.png', dpi=300, bbox_inches='tight')
plt.show()
print("  Figure sauvegardee : 01_signaux_temporels.png")
print()

# --- Questions 2.1 a 2.4 : Identification des signaux ---

print("  Identifiez chaque signal puis repondez :")
print()
print("  2.1 Signal correspondant au NRZ Unipolaire :")
print("      Signal numero    = __________")
print("      Bit 0 -> Tension = __________ V")
print("      Bit 1 -> Tension = __________ V")
print("      Duree de chaque niveau = __________ ms")
print("      Transition 1->0 : OUI / NON")
print("      Transition 0->1 : OUI / NON")
print()
print("  2.2 Signal correspondant au NRZ Bipolaire :")
print("      Signal numero    = __________")
print("      Bit 0 -> Tension = __________ V")
print("      Bit 1 -> Tension = __________ V")
print("      Amplitude crete-a-crete = __________ V")
print("      Sequence '0 1 0 1' : tensions = __________")
print("      Moyenne = __________ V")
print()
print("  2.3 Signal correspondant au RZ Unipolaire :")
print("      Signal numero    = __________")
print("      Bit 0 -> __________ V pendant __________ ms")
print("      Bit 1 -> __________ V pendant __________ ms,")
print("               puis 0 V pendant __________ ms")
print("      Duree impulsion = Tb / __________")
print()
print("  2.4 Signal correspondant au Manchester :")
print("      Signal numero    = __________")
print("      Bit 0 -> Transition : __________ -> __________")
print("      Bit 1 -> Transition : __________ -> __________")
print("      Moment de transition : Debut / Milieu / Fin du bit")
print("      Transitions/bit = __________")
print()

# ============================================================
#  MANIPULATION 3 : Analyse Spectrale (Questions 3.1, 3.2)
# ============================================================

print("=" * 60)
print("  MANIPULATION 3 : Analyse Spectrale")
print("=" * 60)
print()

# --- Calcul DSP (Methode de Welch) ---

nperseg = 1024

sigs_dsp_list = [sig_1_dsp, sig_2_dsp, sig_3_dsp, sig_4_dsp]
noms_dsp = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4']

psds_all = []
for sig in sigs_dsp_list:
    f_w, psd = welch(sig, fs=Fs, nperseg=nperseg, noverlap=nperseg//2,
                      detrend=False)
    psds_all.append(psd)

# --- Figure 02_dsp_comparaison.png ---

max_global = max([np.max(p) for p in psds_all])

psd_1_db = 10 * np.log10(psds_all[0] / max_global + 1e-12)
psd_2_db = 10 * np.log10(psds_all[1] / max_global + 1e-12)
psd_3_db = 10 * np.log10(psds_all[2] / max_global + 1e-12)
psd_4_db = 10 * np.log10(psds_all[3] / max_global + 1e-12)

plt.figure(figsize=(12, 6))

plt.plot(f_w/1000, psd_1_db, color=couleur1, linewidth=2,
         label='Signal 1 (Rouge)')
plt.plot(f_w/1000, psd_2_db, color=couleur2, linewidth=2,
         label='Signal 2 (Vert)')
plt.plot(f_w/1000, psd_3_db, color=couleur3, linewidth=2,
         label='Signal 3 (Bleu)')
plt.plot(f_w/1000, psd_4_db, color=couleur4, linewidth=2,
         label='Signal 4 (Violet)')

plt.plot([Rb/1000, Rb/1000], [-40, 3], 'k--', linewidth=1)
plt.text(Rb/1000 + 0.02, -2, 'Rb', fontsize=10, fontweight='bold')

plt.plot([2*Rb/1000, 2*Rb/1000], [-40, 3], 'k:', linewidth=1)
plt.text(2*Rb/1000 + 0.02, -2, '2Rb', fontsize=10, fontweight='bold')

plt.xlabel('Frequence (kHz)', fontsize=12)
plt.ylabel('DSP (dB/Hz)', fontsize=12)
plt.title('Densite Spectrale de Puissance - Comparaison des Codes',
          fontweight='bold', fontsize=13)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True)
plt.xlim([0, 4])
plt.ylim([-40, 3])

plt.savefig('02_dsp_comparaison.png', dpi=300, bbox_inches='tight')
plt.show()
print("  Figure sauvegardee : 02_dsp_comparaison.png")
print()

# --- Question 3.1 : Identification des courbes DSP ---

print("  3.1 Observation DSP")
print("      Identification des courbes :")
print("      Rouge  (Signal 1) = __________")
print("      Vert   (Signal 2) = __________")
print("      Bleu   (Signal 3) = __________")
print("      Violet (Signal 4) = __________")
print()

# --- Question 3.2 : Tableau d'analyse spectral ---

print("  3.2 Tableau d'Analyse Spectral")
print()

# Mesures automatiques
print("  --- Niveaux de tension ---")
noms = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4']
sigs = [sig_1, sig_2, sig_3, sig_4]
for k in range(4):
    print(f"      {noms[k]} : min = {np.min(sigs[k]):+.1f} V, max = {np.max(sigs[k]):+.1f} V")

print()
print("  --- Composante DC ---")
for k in range(4):
    dc = np.mean(sigs[k])
    if abs(dc) < 0.001:
        print(f"      {noms[k]} : DC = {dc:+.4f} V  --> PAS de composante DC")
    else:
        print(f"      {noms[k]} : DC = {dc:+.4f} V  --> DC PRESENTE")

print()
print("  --- Energie DC sur la DSP ---")
for k in range(4):
    dc_energy = psds_all[k][0]
    if dc_energy < 0.01:
        print(f"      {noms[k]} : DSP(f=0) = {dc_energy:.4f}  --> NEGLIGEABLE")
    else:
        print(f"      {noms[k]} : DSP(f=0) = {dc_energy:.4f}  --> PIC DC PRESENT")

print()
print("  --- Transitions ---")
for k in range(4):
    trans = np.sum(np.diff(sigs[k]) != 0)
    print(f"      {noms[k]} : {trans} transitions, {trans/N_bits:.2f} transitions/bit")

print()
print("  --- Bande passante a -3dB ---")
bw_vals = []
for k in range(4):
    psd_db_k = 10 * np.log10(psds_all[k] / np.max(psds_all[k]) + 1e-12)
    idx = np.where(psd_db_k >= -3)[0]
    if len(idx) > 0:
        bw = f_w[idx[-1]]
    else:
        bw = f_w[-1]
    bw_vals.append(bw)
    print(f"      {noms[k]} : BP(-3dB) = {bw:.1f} Hz")

print()
print("  --- Efficacite spectrale ---")
for k in range(4):
    if bw_vals[k] > 0:
        eff = Rb / bw_vals[k]
    else:
        eff = 0
    print(f"      {noms[k]} : eta = {eff:.3f} bit/s/Hz")

# ============================================================
#  SYNTHESE : Tableau a completer (Question 3.2 du TP)
# ============================================================

print()
print("=" * 60)
print("  SYNTHESE : TABLEAUX A COMPLETER")
print("=" * 60)
print()
print("  Tableau d'Analyse Spectral (Question 3.2) :")
print()
print("  +----------------------+----------+----------+----------+----------+")
print("  | Metrique             | NRZ Uni  | NRZ Bi   | RZ Uni   | Manch.   |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Signal numero        |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Bande passante -3dB  |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Premier zero spectral|          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Energie DC @ f=0     |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Efficacite spectrale |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print()

# ============================================================
#  TABLEAU RECAPITULATIF FINAL
# ============================================================

print("  Tableau Recapitulatif Final :")
print()
print("  +----------------------+----------+----------+----------+----------+")
print("  | Critere              | NRZ Uni  | NRZ Bi   | RZ Uni   | Manch.   |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Signal numero        |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Bande passante       |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | DC presente          |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Transitions/bit      |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Recuperation horloge |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Efficacite spectrale |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")
print("  | Applications         |          |          |          |          |")
print("  +----------------------+----------+----------+----------+----------+")

print()
print("=" * 60)
print("  FICHE REPONSE : IDENTIFICATION DES SIGNAUX")
print("=" * 60)
print()
print("  Signal 1 (Rouge)  = ________________________")
print("  Signal 2 (Vert)   = ________________________")
print("  Signal 3 (Bleu)   = ________________________")
print("  Signal 4 (Violet) = ________________________")
print()
print("  Justifiez chaque identification par au moins 2 arguments.")
print()

print("=== ANALYSE TERMINEE ===")
print("Fichiers generes :")
print("  01_signaux_temporels.png")
print("  02_dsp_comparaison.png")