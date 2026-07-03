import subprocess
import sys
import os


scripts = [
    "tda_enotita_a.py",
    "tda_enotita_b.py",
    "tda_enotita_g.py",
    "tda_enotita_d.py",
    "tda_enotita_e.py"
]

def run_script(script_name):
    print("=" * 80)
    print(f"Εκτέλεση: {script_name}")
    print("=" * 80)
    
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n[ΣΦΑΛΜΑ] Το σενάριο {script_name} απέτυχε με κωδικό επιστροφής {result.returncode}\n")
        sys.exit(result.returncode)
    else:
        print(f"\n[ΕΠΙΤΥΧΙΑ] Το σενάριο {script_name} ολοκληρώθηκε επιτυχώς!\n")

if __name__ == "__main__":
    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"[ΠΡΟΕΙΔΟΠΟΙΗΣΗ] Το αρχείο {script} δεν βρέθηκε στον τρέχοντα κατάλογο.")
            
    print("=" * 80)
    print("Όλα τα ερωτήματα εκτελέστηκαν επιτυχώς!")
    print("=" * 80)
