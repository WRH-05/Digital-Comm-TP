#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP 3 - Modulations Numériques - VERSION ÉTUDIANT
4ème GEI, ELN, USTHB
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class ModulationNumerique:
    def __init__(self, EbN0_dB=10):
        self.Rb = 1000
        self.fc = 4000
        self.fs = 32000
        self.Nbits = 1000
        self.EbN0_dB = EbN0_dB
        self.Ns = int(self.fs / self.Rb)

        np.random.seed(42)
        self.bits = np.random.randint(0, 2, self.Nbits)

        self.t = np.arange(0, self.Nbits/self.Rb, 1/self.fs)
        self.port_cos = np.sqrt(2) * np.cos(2 * np.pi * self.fc * self.t)
        self.port_sin = np.sqrt(2) * np.sin(2 * np.pi * self.fc * self.t)
        self.noms_modulations = ['ASK/OOK', 'BPSK', 'QPSK', '16-QAM', '64-QAM']

    # --- fonctions de modulation (ne pas modifier) ---
    def modulation_1(self):
        symboles = self.bits.copy()
        signal_base = np.repeat(symboles, self.Ns)
        signal_module = signal_base * self.port_cos[:len(signal_base)]
        return signal_module, symboles

    def modulation_2(self):
        symboles = 2 * self.bits - 1
        signal_base = np.repeat(symboles, self.Ns)
        signal_module = signal_base * self.port_cos[:len(signal_base)]
        return signal_module, symboles

    def modulation_3(self):
        if len(self.bits) % 2 != 0:
            bits_paires = self.bits[:-1]
        else:
            bits_paires = self.bits
        bits_reshape = bits_paires.reshape(-1, 2)
        symboles_I = 2 * bits_reshape[:, 0] - 1
        symboles_Q = 2 * bits_reshape[:, 1] - 1
        symboles_complexes = symboles_I + 1j * symboles_Q
        signal_I = np.repeat(symboles_I, self.Ns * 2)
        signal_Q = np.repeat(symboles_Q, self.Ns * 2)
        t_qpsk = self.t[:len(signal_I)]
        port_cos_qpsk = np.sqrt(2) * np.cos(2 * np.pi * self.fc * t_qpsk)
        port_sin_qpsk = np.sqrt(2) * np.sin(2 * np.pi * self.fc * t_qpsk)
        signal_module = signal_I * port_cos_qpsk - signal_Q * port_sin_qpsk
        return signal_module, symboles_complexes

    def modulation_4(self):
        if len(self.bits) % 4 != 0:
            bits_groupes = self.bits[:-(len(self.bits) % 4)]
        else:
            bits_groupes = self.bits
        bits_reshape = bits_groupes.reshape(-1, 4)
        mapping_I = {0: -3, 1: -1, 2: +1, 3: +3}
        mapping_Q = {0: -3, 1: -1, 2: +1, 3: +3}
        symboles_I = np.array([mapping_I[2*bits[0] + bits[1]] for bits in bits_reshape])
        symboles_Q = np.array([mapping_Q[2*bits[2] + bits[3]] for bits in bits_reshape])
        facteur_normalisation = np.sqrt(10)
        symboles_I = symboles_I / facteur_normalisation
        symboles_Q = symboles_Q / facteur_normalisation
        symboles_complexes = symboles_I + 1j * symboles_Q
        signal_I = np.repeat(symboles_I, self.Ns * 4)
        signal_Q = np.repeat(symboles_Q, self.Ns * 4)
        t_qam = self.t[:len(signal_I)]
        port_cos_qam = np.sqrt(2) * np.cos(2 * np.pi * self.fc * t_qam)
        port_sin_qam = np.sqrt(2) * np.sin(2 * np.pi * self.fc * t_qam)
        signal_module = signal_I * port_cos_qam - signal_Q * port_sin_qam
        return signal_module, symboles_complexes

    def modulation_5(self):
        if len(self.bits) % 6 != 0:
            bits_groupes = self.bits[:-(len(self.bits) % 6)]
        else:
            bits_groupes = self.bits
        bits_reshape = bits_groupes.reshape(-1, 6)
        mapping_I = {0: -7, 1: -5, 2: -3, 3: -1, 4: +1, 5: +3, 6: +5, 7: +7}
        mapping_Q = {0: -7, 1: -5, 2: -3, 3: -1, 4: +1, 5: +3, 6: +5, 7: +7}
        symboles_I = np.array([mapping_I[4*bits[0] + 2*bits[1] + bits[2]] for bits in bits_reshape])
        symboles_Q = np.array([mapping_Q[4*bits[3] + 2*bits[4] + bits[5]] for bits in bits_reshape])
        facteur_normalisation = np.sqrt(42)
        symboles_I = symboles_I / facteur_normalisation
        symboles_Q = symboles_Q / facteur_normalisation
        symboles_complexes = symboles_I + 1j * symboles_Q
        signal_I = np.repeat(symboles_I, self.Ns * 6)
        signal_Q = np.repeat(symboles_Q, self.Ns * 6)
        t_qam = self.t[:len(signal_I)]
        port_cos_qam = np.sqrt(2) * np.cos(2 * np.pi * self.fc * t_qam)
        port_sin_qam = np.sqrt(2) * np.sin(2 * np.pi * self.fc * t_qam)
        signal_module = signal_I * port_cos_qam - signal_Q * port_sin_qam
        return signal_module, symboles_complexes

    # --- fonctions d'affichage (à compléter) ---
    def ajouter_bruit_symboles(self, symboles, EbN0_dB=None):
        if EbN0_dB is None:
            EbN0_dB = self.EbN0_dB
        EbN0 = 10**(EbN0_dB / 10)
        Es = np.mean(np.abs(symboles)**2)
        M = len(np.unique(symboles))
        Eb = Es / np.log2(M)
        N0 = Eb / EbN0
        if np.isreal(symboles).all():
            bruit = np.sqrt(N0/2) * np.random.randn(len(symboles))
        else:
            bruit = np.sqrt(N0/2) * (np.random.randn(len(symboles)) + 
                                    1j * np.random.randn(len(symboles)))
        return symboles + bruit

    def tracer_signaux_temporels(self, signaux, n_bits=20):
        fig, axs = plt.subplots(5, 1, figsize=(12, 10))
        # ### A COMPLETER ###
        modulations = self.noms_modulations
        colors = ['blue', 'red', 'green', 'purple', 'orange']

        n_echantillons = n_bits * self.Ns

        for i, (sig, mod, color) in enumerate(zip(signaux, modulations, colors)):
            sig_tronque = sig[:min(n_echantillons, len(sig))]
            t_tronque = self.t[:len(sig_tronque)]
            axs[i].plot(t_tronque * 1000, sig_tronque, color=color, linewidth=1)
            axs[i].set_ylabel(f'{mod}\nAmplitude', fontsize=10)
            axs[i].grid(True, alpha=0.3)
            axs[i].set_xlim([0, n_bits/self.Rb * 1000])
            if i == len(modulations) - 1:
                axs[i].set_xlabel('Temps (ms)')
            else:
                axs[i].set_xticklabels([])

        # ### A COMPLETER ###
        plt.suptitle('Signaux temporels des cinq modulations numériques', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('01_signaux_modules.png', dpi=300, bbox_inches='tight')
        plt.show()

    def tracer_constellations(self, symboles_list, EbN0_dB_list=[0, 5, 10, 15]):
        fig, axs = plt.subplots(len(symboles_list), len(EbN0_dB_list), 
                               figsize=(15, 12))
        # ### A COMPLETER ###
        modulations = self.noms_modulations
        colors = ['blue', 'red', 'green', 'purple', 'orange']

        if len(symboles_list) == 1:
            axs = axs.reshape(1, -1)

        for i, (symb, mod, color) in enumerate(zip(symboles_list, modulations, colors)):
            if symb is None:
                continue
            for j, EbN0_dB in enumerate(EbN0_dB_list):
                symboles_bruites = self.ajouter_bruit_symboles(symb, EbN0_dB)
                if mod in ['ASK/OOK', 'BPSK']:
                    axs[i, j].plot(np.real(symboles_bruites), np.zeros_like(symboles_bruites), 
                                   marker='.', linestyle='', color=color, markersize=2, alpha=0.6)
                    axs[i, j].set_ylim([-2, 2])
                    points_ideaux = np.unique(np.real(symb))
                    axs[i, j].plot(points_ideaux, np.zeros_like(points_ideaux), 
                                   'ko', markersize=6, markerfacecolor='none')
                else:
                    axs[i, j].plot(np.real(symboles_bruites), np.imag(symboles_bruites), 
                                   marker='.', linestyle='', color=color, markersize=2, alpha=0.6)
                    axs[i, j].set_aspect('equal')
                    axs[i, j].set_ylim([-1.5, 1.5])
                    axs[i, j].set_xlim([-1.5, 1.5])
                    points_ideaux = symb
                    axs[i, j].plot(np.real(points_ideaux), np.imag(points_ideaux), 
                                   'ko', markersize=4, markerfacecolor='none')
                if i == 0:
                    axs[i, j].set_title(f'Eb/N0 = {EbN0_dB} dB', fontsize=12, fontweight='bold')
                if j == 0:
                    etiquette_axe = 'Amplitude' if mod in ['ASK/OOK', 'BPSK'] else 'Q'
                    axs[i, j].set_ylabel(f'{mod}\n{etiquette_axe}', fontsize=10)
                if i == len(symboles_list) - 1:
                    axs[i, j].set_xlabel('I')
                axs[i, j].grid(True, alpha=0.3)
                axs[i, j].axhline(y=0, color='k', linestyle='-', alpha=0.5)
                axs[i, j].axvline(x=0, color='k', linestyle='-', alpha=0.5)

        # ### A COMPLETER ###
        plt.suptitle('Constellations bruitées des cinq modulations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('02_constellations_bruit.png', dpi=300, bbox_inches='tight')
        plt.show()

    def tracer_comparaison_bruit(self, symboles_list):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        # ### A COMPLETER ###
        modulations = self.noms_modulations
        EbN0_values = [0, 5, 10, 15, 20]

        for i, (symb, mod) in enumerate(zip(symboles_list, modulations)):
            if i >= len(axs) - 1:
                break
            for EbN0_dB in EbN0_values:
                symboles_bruites = self.ajouter_bruit_symboles(symb, EbN0_dB)
                if mod in ['ASK/OOK', 'BPSK']:
                    axs[i].plot(np.real(symboles_bruites), np.zeros_like(symboles_bruites), 
                               marker='.', linestyle='', markersize=2, alpha=0.6,
                               label=f'{EbN0_dB} dB' if EbN0_dB == EbN0_values[0] else "")
                else:
                    axs[i].plot(np.real(symboles_bruites), np.imag(symboles_bruites), 
                               marker='.', linestyle='', markersize=1, alpha=0.6,
                               label=f'{EbN0_dB} dB' if EbN0_dB == EbN0_values[0] else "")
            axs[i].set_title(mod, fontsize=12, fontweight='bold')
            axs[i].grid(True, alpha=0.3)
            axs[i].set_xlabel('I')
            axs[i].set_ylabel('Amplitude' if mod in ['ASK/OOK', 'BPSK'] else 'Q')
            if mod in ['ASK/OOK', 'BPSK']:
                axs[i].set_ylim([-2, 2])
            else:
                axs[i].set_aspect('equal')
                axs[i].set_ylim([-1.5, 1.5])
                axs[i].set_xlim([-1.5, 1.5])

        axs[-1].axis('off')
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='gray', markersize=8, label=f'Eb/N0 = {dB} dB')
                          for dB in EbN0_values]
        axs[-1].legend(handles=legend_elements, loc='center', title="Niveaux de bruit")

        # ### A COMPLETER ###
        plt.suptitle('Comparaison de l\'effet du bruit sur les cinq modulations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('03_comparaison_bruit.png', dpi=300, bbox_inches='tight')
        plt.show()

    def afficher_parametres(self):
        print("=" * 60)
        print("PARAMÈTRES DE SIMULATION - MODULATIONS NUMÉRIQUES")
        print("=" * 60)
        print(f"Débit binaire (Rb): {self.Rb} Hz")
        print(f"Fréquence porteuse (fc): {self.fc} Hz")
        print(f"Fréquence d'échantillonnage (fs): {self.fs} Hz")
        print(f"Rapport fs/fc: {self.fs / self.fc:.1f}")
        print(f"Nombre de bits: {self.Nbits}")
        print(f"Durée d'un bit (Tb): {1000 / self.Rb:.1f} ms")
        # ### A COMPLETER ###
        print(f"Rapport Eb/N0: {self.EbN0_dB} dB")
        print(f"Nombre d'échantillons par bit: {self.Ns}")
        print("\nPremiers 20 bits de la séquence:")
        print(" ".join(map(str, self.bits[:20])))
        print("=" * 60)

def main():
    print("Début de la simulation des modulations numériques...")

    for EbN0_dB in [10]:
        print(f"\n{'='*50}")
        print(f"Simulation avec Eb/N0 = {EbN0_dB} dB")
        print(f"{'='*50}")

        mod = ModulationNumerique(EbN0_dB=EbN0_dB)
        mod.afficher_parametres()

        print("\nGénération des signaux modulés...")

        signal_1, symboles_1 = mod.modulation_1()
        symboles_1_1d = 2 * mod.bits - 1
        signal_2, symboles_2 = mod.modulation_2()
        signal_3, symboles_3 = mod.modulation_3()
        signal_4, symboles_4 = mod.modulation_4()
        signal_5, symboles_5 = mod.modulation_5()

        signaux = [signal_1, signal_2, signal_3, signal_4, signal_5]
        symboles = [symboles_1_1d, symboles_2, symboles_3, symboles_4, symboles_5]

        if EbN0_dB == 10:
            print("Génération des figures temporelles...")
            mod.tracer_signaux_temporels(signaux)

    mod_default = ModulationNumerique(EbN0_dB=10)
    _, symboles_1_final = mod_default.modulation_1()
    symboles_1_1d_final = 2 * mod_default.bits - 1
    _, symboles_2_final = mod_default.modulation_2()
    _, symboles_3_final = mod_default.modulation_3()
    _, symboles_4_final = mod_default.modulation_4()
    _, symboles_5_final = mod_default.modulation_5()

    symboles_finaux = [symboles_1_1d_final, symboles_2_final, symboles_3_final,
                       symboles_4_final, symboles_5_final]

    print("\nGénération des diagrammes de constellation avec bruit...")
    mod_default.tracer_constellations(symboles_finaux)
    mod_default.tracer_comparaison_bruit(symboles_finaux)

    # ### A COMPLETER ###
    print("\n=== IDENTIFICATION DES MODULATIONS ===")
    print("Modulation 1 : ASK/OOK (nombre de points : 2)")
    print("Modulation 2 : BPSK (nombre de points : 2)")
    print("Modulation 3 : QPSK (nombre de points : 4)")
    print("Modulation 4 : 16-QAM (nombre de points : 16)")
    print("Modulation 5 : 64-QAM (nombre de points : 64)")

    print("\nSimulation terminée avec succès!")
    print("Figures sauvegardées:")
    print("  - 01_signaux_modules.png")
    print("  - 02_constellations_bruit.png")
    print("  - 03_comparaison_bruit.png")

if __name__ == "__main__":
    main()