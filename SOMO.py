#!/usr/bin/env python3
import re, argparse, math, sys

kB = 1.380649e-23  # J/K

def parse_grpy_output(filename):
    """
    Parse SoMo/GRPY hydrodynamic output and return:
      Dt_m2s, Dr_sinv, Rg_m (or None), Rh_trans_m (or None), eta_Pas (or None), T_kelvin (or None)
    """
    Dt = None       # translational diffusion coefficient [m^2/s]
    Dr = None       # rotational diffusion coefficient [s^-1] (orientation average)
    Rg = None       # radius of gyration [m]
    Rh_trans = None # translational Stokes radius [m]
    eta = None      # viscosity [Pa*s]
    T_K = None      # absolute temperature [K] if detectable (optional)

    # Regexes to catch common variants in SoMo/GRPY reports
    re_Dt = re.compile(r"Translational diffusion coefficient:\s*([0-9.Ee+-]+)\s*\[cm\^2/s\]")
    re_Dr = re.compile(r"Rotational diffusion coefficient:\s*([0-9.Ee+-]+)\s*\[s\^-1\]")
    re_Rg = re.compile(r"RADIUS OF GYRATION\s*=\s*([0-9.Ee+-]+)\s*\[nm\]")
    re_Rh = re.compile(r"TRANSLATIONAL STOKES'? RADIUS\s*=\s*([0-9.Ee+-]+)\s*\[nm\]", re.IGNORECASE)

    # Viscosity lines appear in a few styles:
    re_eta_used   = re.compile(r"Used\s+solvent viscosity\s+([0-9.Ee+-]+)\s*\[cP\]", re.IGNORECASE)
    re_eta_solvent= re.compile(r"Solvent viscosity\s*\(cP\)\s*:\s*([0-9.Ee+-]+)", re.IGNORECASE)

    # Temperature lines (either in C or K depending on context)
    re_T_c = re.compile(r"Temperature\s*:\s*([0-9.Ee+-]+)\s*$")           # e.g. "Temperature: 20.00"
    re_T_k = re.compile(r"Temperature\s*:\s*([0-9.Ee+-]+)\s*\[K\]")       # rarer variant

    with open(filename, 'r') as f:
        for line in f:
            m = re_Dt.search(line)
            if m:
                Dt = float(m.group(1)) * 1e-4  # cm^2/s -> m^2/s
                continue
            m = re_Dr.search(line)
            if m and Dr is None:
                Dr = float(m.group(1))         # s^-1
                continue
            m = re_Rg.search(line)
            if m:
                Rg = float(m.group(1)) * 1e-9  # nm -> m
                continue
            m = re_Rh.search(line)
            if m:
                Rh_trans = float(m.group(1)) * 1e-9  # nm -> m
                continue
            m = re_eta_used.search(line) or re_eta_solvent.search(line)
            if m:
                eta = float(m.group(1)) * 1e-3  # cP -> Pa*s
                continue
            m = re_T_k.search(line)
            if m:
                T_K = float(m.group(1))
                continue
            m = re_T_c.search(line)
            if m:
                # Most SoMo lines list Celsius; convert to Kelvin if value looks like 0–100-ish
                Tc = float(m.group(1))
                if 0.0 <= Tc <= 150.0:
                    T_K = Tc + 273.15
                else:
                    # Already Kelvin-like
                    T_K = Tc
                continue

    if Dt is None or Dr is None:
        raise ValueError("Could not parse Dt or Dr from file. Check file formatting or enable rotational outputs (GRPY/ZENO).")

    return Dt, Dr, Rg, Rh_trans, eta, T_K

def compute_frictions(Dt, Dr, T_K=298.0):
    """Compute translational and rotational frictions from diffusion coefficients."""
    A = kB*T_K / Dt   # N*s/m
    B = kB*T_K / Dr   # N*s*m
    return A, B

def rh_from_dt(Dt, eta, T_K):
    """Hydrodynamic (Stokes) radius from Dt via Stokes-Einstein: Rh = kBT/(6*pi*eta*Dt)."""
    return (kB*T_K) / (6.0*math.pi*eta*Dt)

def predict_D1(A, B, Roc_nm, pitch_nm=3.4, T_K=298.0):
    """Predict D1 using BBX/BBXB helical friction formula."""
    b = (pitch_nm*1e-9)/(2*math.pi)  # m
    Roc = Roc_nm*1e-9                # m
    zeta_tot = A + (B + A*Roc**2)/b**2
    D1 = kB*T_K / zeta_tot
    return D1, zeta_tot

def main():
    ap = argparse.ArgumentParser(description="Parse GRPY/SoMo output and compute hydrodynamic radii and BBX/BBXB D1.")
    ap.add_argument("txt", help="GRPY output text file")
    ap.add_argument("--temp", type=float, default=None, help="Override Temperature (K) used for A,B and D1. If omitted, use T from file if present, else 298 K.")
    ap.add_argument("--Roc", type=float, default=2.5, help="Offset distance R_OC (nm)")
    ap.add_argument("--pitch", type=float, default=3.4, help="DNA pitch per turn (nm)")
    ap.add_argument("--eta", type=float, default=None, help="Override viscosity (Pa*s) for computing Rh if file lacks viscosity.")
    args = ap.parse_args()

    Dt, Dr, Rg, Rh_reported, eta_file, T_file = parse_grpy_output(args.txt)

    # Choose temperature for friction/D1 calculations
    if args.temp is not None:
        T_K = float(args.temp)
    elif T_file is not None:
        T_K = T_file
    else:
        T_K = 298.0  # default

    # Compute A,B and D1
    A, B = compute_frictions(Dt, Dr, T_K)
    D1, zeta_tot = predict_D1(A, B, args.Roc, pitch_nm=args.pitch, T_K=T_K)

    # Determine viscosity for Rh_from_Dt if needed
    eta_used = None
    eta_source = None
    if eta_file is not None:
        eta_used = eta_file
        eta_source = "from file"
    elif args.eta is not None:
        eta_used = float(args.eta)
        eta_source = "from --eta"
    else:
        eta_used = 1.0e-3
        eta_source = "default 1.00e-3 Pa*s"

    Rh_computed = rh_from_dt(Dt, eta_used, T_K)

    # ---- Reporting ----
    print(f"Parsed file: {args.txt}")
    print(f"Temperature used (K)     = {T_K:.2f} " + ("(override)" if args.temp is not None else "(from file)" if T_file is not None else "(default 298 K)"))
    print(f"R_OC                     = {args.Roc:.2f} nm")
    print(f"DNA pitch                = {args.pitch:.2f} nm")
    print()
    print(f"Translational diffusion Dt     = {Dt:.3e} m^2/s")
    print(f"Rotational diffusion Dr        = {Dr:.3e} 1/s")
    if Rg is not None:
        print(f"Radius of gyration Rg          = {Rg*1e9:.2f} nm")
    # Hydrodynamic radius reporting
    if Rh_reported is not None:
        print(f"Translational Stokes radius Rh (SoMo-reported) = {Rh_reported*1e9:.3f} nm")
    else:
        print(f"Translational Stokes radius Rh (SoMo-reported) = n/a")
    print(f"Translational Stokes radius Rh (computed from Dt, eta={eta_used:.2e} Pa*s; {eta_source}) = {Rh_computed*1e9:.3f} nm")
    print()
    print(f"Translational friction A = {A:.3e} N*s/m")
    print(f"Rotational friction B    = {B:.3e} N*s*m")
    print(f"Total friction zeta_tot  = {zeta_tot:.3e} N*s/m")
    print(f"Predicted D1             = {D1:.3e} m^2/s")

if __name__ == "__main__":
    main()
