# ============================================================================
# TP 2 : FILTRAGE DE NYQUIST ET TRANSMISSION EN BANDE LIMITÉE
# ============================================================================
# Établissement : USTHB - Faculté de Genie electrique
# Département : 4ème année GEI, ELN
# Année universitaire : 2025-2026
# Chargé de TP : Mr. MEFTAH
# ============================================================================
#
# INSTRUCTIONS :
# 1. Version completee : toutes les zones precedemment masquees sont renseignees
# 2. Les legendes α et instants d'echantillonnage sont explicites
# 3. Les questions techniques du code sont repondues dans les commentaires
# 4. Les tableaux numeriques se remplissent automatiquement a l'execution
#
# FICHIER : filtrage_nyquist_etudiant.py
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FiltrageNyquist:
    def __init__(self, Rb=1000, fs=8000, Nbits=200):
        self.Rb = Rb
        self.fs = fs
        self.Nbits = Nbits
        self.Tb = 1/Rb
        # Relation d'echantillonnage : Ns = fs * Tb (nombre d'echantillons par bit)
        self.Ns = int(fs * self.Tb)
        if self.Ns == 0:
            self.Ns = 1
        self.t = np.arange(0, Nbits * self.Tb, 1/fs)
        self.output_dir = Path(__file__).resolve().parent / 'output'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sauvegarder_figure(self, fig, nom_fichier):
        chemin = self.output_dir / nom_fichier
        fig.savefig(chemin, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardee: {chemin}")
        
    def sinc(self, x):
        # Reponse impulsionnelle ideale d'un filtre passe-bas de Nyquist en bande de base.
        x = np.asarray(x, dtype=float)
        result = np.ones_like(x)
        mask = np.abs(x) > 1e-10
        result[mask] = np.sin(np.pi * x[mask]) / (np.pi * x[mask])
        return result
    
    def filtre_cosinus_sureleve(self, alpha, L=10):
        # h(t)=sinc(t/Tb)*cos(pi*alpha*t/Tb)/(1-(2*alpha*t/Tb)^2)/Tb, avec traitement des singularites.
        N_pts = 2 * L * self.Ns + 1
        t_imp = np.linspace(-L*self.Tb, L*self.Tb, N_pts)
        h = np.zeros(N_pts, dtype=float)
        
        for i, t_val in enumerate(t_imp):
            t_norm = t_val / self.Tb
            
            if abs(t_val) < 1e-12:
                h[i] = 1.0 / self.Tb
            elif alpha > 1e-10 and abs(abs(2*alpha*t_norm) - 1.0) < 1e-10:
                h[i] = (np.pi / (4 * self.Tb)) * self.sinc(1/(2*alpha))
            else:
                sinc_val = self.sinc(t_norm)
                cos_val = np.cos(np.pi * alpha * t_norm)
                den = 1 - (2*alpha*t_norm)**2
                
                if abs(den) < 1e-10:
                    h[i] = (np.pi / (4 * self.Tb)) * self.sinc(1/(2*alpha))
                else:
                    h[i] = (sinc_val * cos_val / den) / self.Tb
                    # Pour alpha=0, cos(...) = 1 et le denominateur vaut 1 : on retrouve un sinc pur.
        
        h = h / np.sqrt(np.sum(h**2))
        return t_imp, h
    
    def reponse_frequentielle(self, h, Nfft=2048):
        # Evite log(0) et stabilise l'affichage en dB sur les zones de gain quasi nul.
        H = np.fft.fft(h, Nfft)
        H = np.fft.fftshift(H)
        f = np.fft.fftshift(np.fft.fftfreq(Nfft, 1/self.fs))
        return f, 20*np.log10(np.abs(H) + 1e-12)
    
    def generer_sequence(self, seed=None):
        # Mapping binaire antipodal de type BPSK : 0 -> -1, 1 -> +1.
        if seed is not None:
            np.random.seed(seed)
        bits = np.random.randint(0, 2, self.Nbits)
        symbols = 2*bits - 1
        return bits, symbols.astype(float)
    
    def sur_echantillonnage(self, symbols):
        signal_sur = np.zeros(len(symbols) * self.Ns)
        signal_sur[::self.Ns] = symbols
        return signal_sur
    
    def filtrer_signal(self, signal_in, h):
        return np.convolve(signal_in, h, mode='same')
    
    def diagramme_oeil(self, signal_rx, alpha):
        # Superposition de trames de 2Tb pour visualiser ouverture d'oeil, jitter tolere et IES.
        N_oeil = 2 * self.Ns
        
        if len(signal_rx) < N_oeil + 20*self.Ns:
            print("Signal trop court pour le diagramme de l'œil")
            return
            
        plt.figure(figsize=(12, 6))
        
        start_idx = 15 * self.Ns
        count = 0
        
        for i in range(start_idx, len(signal_rx) - N_oeil, self.Ns):
            segment = signal_rx[i:i + N_oeil]
            temps_oeil = np.linspace(-1, 1, len(segment))
            plt.plot(temps_oeil, segment, 'b-', alpha=0.15, linewidth=0.5)
            count += 1
            if count > 100:
                break
        
        plt.axvline(x=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Instant optimal (t = kTb)')
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.title(f'Diagramme de l\'oeil du signal recu - α = {alpha}')
        plt.xlabel('Temps normalise (en Tb)')
        plt.ylabel('Amplitude (V)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-1, 1)
        plt.tight_layout()
        alpha_tag = str(alpha).replace('.', 'p')
        self.sauvegarder_figure(plt.gcf(), f'diagramme_oeil_alpha_{alpha_tag}.png')
        plt.show()
    
    def calcul_ouverture_ies(self, signal_rx, decalage_echantillons, symbols):
        # IES (ISI) mesure l'interference inter-symboles ; critere de Nyquist ideal => IES nulle aux instants kTb.
        start_sym = 20
        end_sym = min(self.Nbits - 20, len(signal_rx) // self.Ns - 5)
        
        if end_sym <= start_sym + 10:
            return 0, 1.0, np.array([])
        
        indices = []
        for k in range(start_sym, end_sym):
            idx = k * self.Ns + decalage_echantillons
            if 0 <= idx < len(signal_rx):
                indices.append(idx)
        
        if len(indices) < 10:
            return 0, 1.0, np.array([])
        
        indices = np.array(indices)
        echantillons = signal_rx[indices]
        symboles_ref = symbols[start_sym:start_sym + len(echantillons)]
        
        mask_pos = symboles_ref > 0
        mask_neg = symboles_ref < 0
        
        ech_pos = echantillons[mask_pos]
        ech_neg = echantillons[mask_neg]
        
        if len(ech_pos) < 5 or len(ech_neg) < 5:
            return 0, 1.0, echantillons
        
        min_pos = np.min(ech_pos)
        max_neg = np.max(ech_neg)
        ouverture = max(0, min_pos - max_neg)
        
        max_pos = np.max(ech_pos)
        min_neg = np.min(ech_neg)
        plage_totale = max_pos - min_neg
        
        if plage_totale > 1e-10:
            ies = 1.0 - ouverture / plage_totale
        else:
            ies = 1.0
        
        return ouverture, ies, echantillons
    
    def calcul_marge_bruit(self, ouverture):
        # Plus l'ouverture est grande, plus la decision tolere du bruit additif avant erreur de symbole.
        if ouverture <= 1e-10:
            return -40.0
        return 20 * np.log10(ouverture / 2)
    
    def calcul_penalite_snr(self, ouv_actuelle, ouv_reference):
        if ouv_reference <= 1e-10 or ouv_actuelle <= 1e-10:
            return float('inf')
        return 20 * np.log10(ouv_reference / ouv_actuelle)
    
    def analyser_alpha(self, alphas=[0.25, 0.5, 0.75]):
        results = {}
        bits, symbols = self.generer_sequence(seed=42)
        
        for alpha in alphas:
            print(f"\n{'='*50}")
            print(f"Analyse pour α = {alpha}")
            print('='*50)
            
            try:
                t_imp, h = self.filtre_cosinus_sureleve(alpha)
                
                signal_sur = self.sur_echantillonnage(symbols)
                signal_tx = self.filtrer_signal(signal_sur, h)
                signal_canal = signal_tx.copy()
                
                # Filtrage adapte (matched filter) : meme filtre Tx/Rx pour maximiser SNR et respecter Nyquist global.
                signal_rx = self.filtrer_signal(signal_canal, h)
                
                self.diagramme_oeil(signal_rx, alpha)
                
                instant_optimal = 0 
                ouverture, ies, echantillons = self.calcul_ouverture_ies(
                    signal_rx, instant_optimal, symbols
                )
                
                marge_db = self.calcul_marge_bruit(ouverture)
                
                f, H_dB = self.reponse_frequentielle(h)
                H_linear = 10**(H_dB/20)
                H_max = np.max(H_linear)
                # 0.707 = 1/sqrt(2), soit le niveau -3 dB de reference pour la bande passante.
                idx_3db = np.where(H_linear >= 0.707 * H_max)[0]
                BW = f[idx_3db[-1]] - f[idx_3db[0]] if len(idx_3db) > 0 else 0
                
                print(f"Ouverture de l'œil: {ouverture:.4f} V")
                print(f"IES mesuré: {ies:.4f}")
                print(f"Marge au bruit: {marge_db:.2f} dB")
                print(f"Bande passante à -3dB: {BW:.1f} Hz")
                
                results[alpha] = {
                    'h': h,
                    't_imp': t_imp,
                    'signal_rx': signal_rx,
                    'symbols': symbols,
                    'ies': ies,
                    'ouverture': ouverture,
                    'marge_db': marge_db,
                    'bande_passante': BW,
                    'echantillons': echantillons
                }
                
            except Exception as e:
                print(f"Erreur pour alpha={alpha}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def analyser_synchronisation(self, alpha=0.5):
        print(f"\n{'='*60}")
        print(f"ANALYSE DE LA SYNCHRONISATION (α = {alpha})")
        print('='*60)
        
        try:
            t_imp, h = self.filtre_cosinus_sureleve(alpha)
            bits, symbols = self.generer_sequence(seed=123)
            signal_sur = self.sur_echantillonnage(symbols)
            signal_tx = self.filtrer_signal(signal_sur, h)
            signal_rx = self.filtrer_signal(signal_tx, h)
            
            # INSTANTS MASQUÉS POUR L'ÉTUDIANT
            decalages = [0, self.Ns//4, self.Ns//2, 3*self.Ns//4]
            labels = ['Instant = 0 (kTb)', 'Instant = Tb/4', 'Instant = Tb/2', 'Instant = 3Tb/4']
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            ouv_ref, _, _ = self.calcul_ouverture_ies(signal_rx, 0, symbols)
            
            print("\n--- Résultats par instant d'échantillonnage ---")
            print(f"{'Instant':<20} {'Ouverture (V)':<15} {'IES':<12} {'Pénalité (dB)':<15}")
            print("-" * 62)
            
            for i, (decalage, label) in enumerate(zip(decalages, labels)):
                ouverture, ies, echantillons = self.calcul_ouverture_ies(
                    signal_rx, decalage, symbols
                )
                
                if len(echantillons) == 0:
                    axes[i].set_title(f'{label} - Pas de données')
                    continue
                
                penalite = self.calcul_penalite_snr(ouverture, ouv_ref)
                
                pen_str = f"{penalite:.2f}" if penalite != float('inf') else "∞"
                print(f"{label:<20} {ouverture:<15.4f} {ies:<12.4f} {pen_str:<15}")
                
                n_plot = min(50, len(echantillons))
                axes[i].plot(echantillons[:n_plot], 'bo-', alpha=0.7, markersize=3)
                axes[i].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Seuil +1 (BPSK)')
                axes[i].axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='Seuil -1 (BPSK)')
                axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                # Quand on s'eloigne de l'instant optimal, l'IES augmente et l'ouverture de l'oeil diminue.
                axes[i].set_title(f'Echantillonnage et sensibilite a l\'IES - {label}\nOuverture: {ouverture:.3f} V, IES: {ies:.3f}')
                axes[i].set_xlabel('Indice de symbole')
                axes[i].set_ylabel('Amplitude echantillonnee (V)')
                axes[i].set_ylim(-2, 2)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(loc='upper right', fontsize=7)
            
            plt.suptitle(f'Impact de la synchronisation sur les echantillons - α = {alpha}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            alpha_tag = str(alpha).replace('.', 'p')
            self.sauvegarder_figure(plt.gcf(), f'analyse_synchronisation_alpha_{alpha_tag}.png')
            plt.show()
            
            # L'instant du décalage est masqué
            print(f"\n--- Analyse spécifique pour décalage à Tb/4 ---")
            ouv_opt, ies_opt, _ = self.calcul_ouverture_ies(signal_rx, 0, symbols)
            ouv_tb4, ies_tb4, _ = self.calcul_ouverture_ies(signal_rx, self.Ns//4, symbols)
            penalite_tb4 = self.calcul_penalite_snr(ouv_tb4, ouv_opt)
            
            print(f"Ouverture optimale (kTb): {ouv_opt:.4f} V")
            print(f"Ouverture à Tb/4: {ouv_tb4:.4f} V")
            print(f"Perte d'ouverture: {(1 - ouv_tb4/ouv_opt)*100:.1f} %")
            print(f"Pénalité en SNR: {penalite_tb4:.2f} dB")
            
        except Exception as e:
            print(f"Erreur: {e}")
            import traceback
            traceback.print_exc()
    
    def analyser_robustesse_alpha(self, alphas=[0.25, 0.5, 0.75]):
        print(f"\n{'='*60}")
        print("ANALYSE COMPARATIVE DE LA ROBUSTESSE")
        print('='*60)
        
        print(f"\n{'Alpha':<10} {'Ouv. opt. (V)':<15} {'Ouv. décalé (V)':<15} {'Pénalité (dB)':<15}")
        print("-" * 55)
        
        resultats = {}
        
        for idx, alpha in enumerate(alphas):
            try:
                t_imp, h = self.filtre_cosinus_sureleve(alpha)
                bits, symbols = self.generer_sequence(seed=456)
                signal_sur = self.sur_echantillonnage(symbols)
                signal_tx = self.filtrer_signal(signal_sur, h)
                signal_rx = self.filtrer_signal(signal_tx, h)
                
                ouv_opt, ies_opt, _ = self.calcul_ouverture_ies(signal_rx, 0, symbols)
                ouv_tb4, ies_tb4, _ = self.calcul_ouverture_ies(signal_rx, self.Ns//4, symbols)
                penalite = self.calcul_penalite_snr(ouv_tb4, ouv_opt)
                
                pen_str = f"{penalite:.2f}" if penalite != float('inf') else "∞"
                print(f"{alpha:<10.2f} {ouv_opt:<15.4f} {ouv_tb4:<15.4f} {pen_str:<15}")
                
            except Exception as e:
                print(f"Filtre {idx+1}   Erreur: {e}")
                continue


def main():
    print("="*70)
    print("TP 2 : FILTRAGE DE NYQUIST")
    print("USTHB - 4ème GEI, ELN - Communications Numériques")
    print("="*70)
    
    sim = FiltrageNyquist(Rb=1000, fs=8000, Nbits=200)
    alphas = [0.25, 0.5, 0.75]
    
    print("\n" + "="*70)
    print("MANIPULATION 1 : CARACTÉRISATION DES FILTRES")
    print("="*70)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    colors = ['blue', 'red', 'green']
    
    print(f"\n{'α':<10} {'Bande -3dB (Hz)':<18} {'1er zéro (Hz)':<18} {'Oscillations':<15}")
    print("-" * 65)
    
    for i, alpha in enumerate(alphas):
        t_imp, h = sim.filtre_cosinus_sureleve(alpha)
        f, H_dB = sim.reponse_frequentielle(h)
        
        ax1.plot(t_imp*1000, h, linewidth=2, color=colors[i], label=f'α = {alpha}')
        ax2.plot(f, H_dB, linewidth=2, color=colors[i], label=f'α = {alpha}')
        
        H_linear = 10**(H_dB/20)
        H_max = np.max(H_linear)
        idx_3db = np.where(H_linear >= 0.707 * H_max)[0]
        BW = f[idx_3db[-1]] - f[idx_3db[0]] if len(idx_3db) > 0 else 0
        
        f_positive = f[f > 0]
        H_positive = np.abs(H_linear[f > 0])
        zeros_idx = np.where(H_positive < 0.01 * np.max(H_positive))[0]
        premier_zero = f_positive[zeros_idx[0]] if len(zeros_idx) > 0 else np.nan
        
        sign_changes = np.sum(np.abs(np.diff(np.sign(h))) > 0)
        
        print(f"{alpha:<10} {BW:<18.1f} {premier_zero:<18.1f} {sign_changes:<15}")
    
    # Aux instants t=kTb, les echantillons hors instant utile tendent vers zero (propriete de Nyquist).
    ax1.set_title('Reponse impulsionnelle des filtres en cosinus sureleve')
    ax1.set_xlabel('Temps (ms)')
    ax1.set_ylabel('h(t) normalisee')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    for k in range(-5, 6):
        ax1.axvline(x=k*sim.Tb*1000, color='gray', linestyle=':', alpha=0.4)
    
    # Plus alpha augmente, plus la transition est douce mais la bande passante occupee augmente.
    ax2.set_title('Reponse frequentielle des filtres (base bande)')
    ax2.set_xlabel('Frequence (Hz)')
    ax2.set_ylabel('Gain (dB)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-2000, 2000)
    ax2.set_ylim(-60, 5)
    ax2.axhline(y=-3, color='k', linestyle='--', alpha=0.5, label='Niveau -3 dB')
    
    plt.tight_layout()
    sim.sauvegarder_figure(plt.gcf(), 'manipulation_1_caracterisation_filtres.png')
    plt.show()
    
    print("\n" + "="*70)
    print("MANIPULATION 2 : ANALYSE DE LA CHAÎNE DE TRANSMISSION")
    print("="*70)
    results = sim.analyser_alpha(alphas=alphas)
    
    print("\n" + "="*70)
    print("MANIPULATION 3 : OPTIMISATION ET ROBUSTESSE")
    print("="*70)
    sim.analyser_synchronisation(alpha=0.5)
    sim.analyser_robustesse_alpha(alphas=alphas)

    print(f"\nLes figures ont ete sauvegardees dans: {sim.output_dir}")
    
    print("\n" + "="*70)
    print("TABLEAU RÉCAPITULATIF")
    print("="*70)
    
    print(f"\n{'Métrique':<25} ", end="")
    for alpha in alphas:
        print(f"{f'α = {alpha}':<18}", end="")
    print("\n" + "-" * 79)
    
    for key, name in [('ouverture', 'Ouverture œil (V)'), 
                      ('ies', 'IES mesuré'), 
                      ('marge_db', 'Marge au bruit (dB)'), 
                      ('bande_passante', 'Bande passante (Hz)')]:
        print(f"{name:<25} ", end="")
        for alpha in alphas:
            if alpha in results:
                val = results[alpha][key]
                fmt = ".2f" if key == 'marge_db' or key == 'bande_passante' else ".4f"
                print(f"{val:<18{fmt}}", end="")
            else:
                print(f"{'N/A':<18}", end="")
        print()
    
    print("-" * 79)
    # Recommandation pratique: alpha intermediaire (ex. 0.5) pour compromis bande passante/robustesse.
    
    print("\n" + "="*70)
    print("FIN DU TP 2")
    print("="*70)


if __name__ == "__main__":
    main()