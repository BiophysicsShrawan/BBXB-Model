#!/usr/bin/env python3
import numpy as np, math, argparse, csv, sys
from pathlib import Path

kB = 1.380649e-23  # J/K

# ------------------------ BIOMT parser ------------------------
def parse_biomt_from_pdb(pdb_path):
    """Parse REMARK 350 BIOMT matrices (robust token-based). Returns list of 3x4 transforms."""
    blocks = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("REMARK 350"):
                continue
            if "BIOMT" not in line:
                continue
            tail = line.split("REMARK 350", 1)[1].strip()
            if not tail.startswith("BIOMT"):
                continue
            toks = tail.split()
            tag = toks[0]  # e.g., BIOMT1
            if not tag.startswith("BIOMT") or len(tag) < 6:
                continue
            try:
                row_idx = int(tag[-1]) - 1  # 0,1,2
            except ValueError:
                continue
            serial = 1
            if len(toks) >= 2:
                try:
                    serial = int(float(toks[1]))
                    num_start = 2
                except ValueError:
                    num_start = 1
            else:
                num_start = 1
            needed = toks[num_start:num_start+4]
            if len(needed) < 4:
                # fallback to fixed columns
                try:
                    m1 = float(line[33:40]); m2 = float(line[40:47])
                    m3 = float(line[47:54]); t  = float(line[57:68])
                    needed = [m1, m2, m3, t]
                except Exception:
                    continue
            try:
                row_vals = [float(x) for x in needed]
            except ValueError:
                continue
            if serial not in blocks:
                blocks[serial] = [None, None, None]
            blocks[serial][row_idx] = row_vals

    mats = []
    for s in sorted(blocks.keys()):
        rows = blocks[s]
        if any(r is None for r in rows):
            continue
        M = np.zeros((3,4), float)
        for i in range(3):
            M[i,:3] = rows[i][0:3]
            M[i, 3] = rows[i][3]
        mats.append(M)
    return mats

# ---------------------- PDB coordinates -----------------------
def load_pdb_coords(pdb_file, atom_names=None, chain_filter=None, altloc_preference='A'):
    coords, seen = [], set()
    with open(pdb_file, 'r') as f:
        for line in f:
            if line[0:6].strip() not in ('ATOM', 'HETATM'):
                continue
            atom = line[12:16].strip()
            alt  = line[16].strip()
            ch   = line[21].strip()
            resi = line[22:26].strip()
            icode= line[26].strip()
            if chain_filter and ch not in chain_filter:
                continue
            if atom_names and atom not in atom_names:
                continue
            key = (ch, resi, icode, atom)
            if key in seen:
                continue
            if alt and altloc_preference and alt != altloc_preference:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            seen.add(key)
            coords.append([x, y, z])
    arr = np.array(coords, float) * 1e-10  # Å → m
    if arr.size == 0:
        raise RuntimeError(f"No atoms parsed from {pdb_file}")
    return arr

def apply_biomt(coords, mats):
    if not mats or coords.size == 0:
        return coords
    ang = coords * 1e10
    one = np.ones((ang.shape[0], 1))
    hom = np.hstack([ang, one])  # (N,4)
    out = []
    for M in mats:
        out.append(hom @ M.T)     # (N,3)
    return np.vstack(out) * 1e-10

# ----------------------- Shape & Perrin -----------------------
def gyration_tensor(coords):
    r = coords - coords.mean(axis=0, keepdims=True)
    return (r.T @ r) / r.shape[0]

def shape_params_from_G(G, eps=1e-30):
    vals, _ = np.linalg.eigh(G)
    vals = np.sort(vals)  # λ1 ≤ λ2 ≤ λ3
    tr = float(np.sum(vals)); lam = tr/3.0
    if tr <= eps:
        return vals, 0.0, 0.0
    Delta = 3.0 * np.sum((vals - lam)**2) / (2.0 * tr**2)
    S     = 27.0 * np.prod(vals - lam) / (tr**3)
    # Physical bound for S
    S_bound = 2.0 * (max(Delta, 0.0) ** 1.5)
    if abs(S) > S_bound and S_bound > 0:
        S = math.copysign(S_bound, S)
    return vals, Delta, S

def p_from_S(S):
    if S == 0.0:
        return 1.0
    s = math.copysign((abs(S)/2.0)**(1.0/3.0), S)
    s = max(min(s, 0.999999), -0.499999)
    return math.sqrt((1.0 + 2.0*s) / (1.0 - s))

def Req_from_Rg(Rg, p):
    return math.sqrt(5.0*Rg*Rg / (p*p + 2.0)) * (p ** (-2.0/3.0))

def perrin_factors(p):
    # near-sphere expansions for stability
    if abs(p - 1.0) < 1e-3:
        eps = p - 1.0
        fT  = 1.0 + (1.0/5.0)*eps**2 - (1.0/35.0)*eps**3
        Fax = 1.0 + (2.0/5.0)*eps - (1.0/7.0)*eps**2
        Feq = 1.0 - (1.0/5.0)*eps - (1.0/35.0)*eps**2
        Frot = (Fax + 2.0*Feq) / 3.0
        return fT, Frot
    # exact Perrin forms
    if p > 1.0:
        xi = math.sqrt(p*p - 1.0) / p
        S_perrin = 2.0 * (0.5 * math.log((1+xi)/(1-xi))) / xi  # 2*atanh(xi)/xi
    else:
        xi = math.sqrt(1.0 - p*p) / p
        S_perrin = 2.0 * (math.atan(xi)) / xi
    fT = 2.0 * (p ** (2.0/3.0)) / S_perrin
    denom_ax = 2.0*p*p - S_perrin
    denom_eq = 2.0 - S_perrin*(2.0 - 1.0/(p*p))
    if abs(denom_ax) < 1e-12: denom_ax = math.copysign(1e-12, denom_ax or 1.0)
    if abs(denom_eq) < 1e-12: denom_eq = math.copysign(1e-12, denom_eq or 1.0)
    Fax = (4.0/3.0) * (p*p - 1.0) / denom_ax
    Feq = (4.0/3.0) * ((1.0/(p*p)) - p*p) / denom_eq
    Frot = (Fax + 2.0*Feq) / 3.0
    return fT, Frot

# ---------------------- BBXB-S (spheroid) ---------------------
def bbxb_S_from_coords(coords, Roc_nm, pitch_nm, temp=298.0, eta=1.0e-3):
    Roc = Roc_nm*1e-9
    b   = (pitch_nm*1e-9) / (2.0*np.pi)
    G = gyration_tensor(coords)
    vals, Delta, S = shape_params_from_G(G)
    Rg = math.sqrt(float(np.sum(vals)))
    p  = p_from_S(S)
    Req= Req_from_Rg(Rg, p)
    fT, Frot = perrin_factors(p)
    A = 6.0*np.pi*eta*Req*fT
    B = 8.0*np.pi*eta*(Req**3)*Frot
    zeta = A + (B + A*(Roc**2)) / (b**2)
    D1   = kB*temp / zeta
    # expose common descriptors (for table)
    desc = dict(Rg=Rg, Delta=Delta, S=S, p=p,
                Req=Req, fT=fT, Frot=Frot, A=A, B=B, zeta=zeta)
    return D1, desc

# ---------------------- BBXB-T (triaxial) ---------------------
def bbxb_T_from_coords(coords, Roc_nm, pitch_nm, temp=298.0, eta=1.0e-3):
    """
    Triaxial surrogate via Perrin-averaging:
      - Semi-axes from gyration eigenvalues (uniform ellipsoid: λi = ai^2/5)
      - R_eq = (abc)^(1/3)
      - Three spheroids aligned with principal axes:
            p_a = a/sqrt(bc), p_b = b/sqrt(ac), p_c = c/sqrt(ab)
      - Average Perrin factors: fT_bar = (fT(p_a)+fT(p_b)+fT(p_c))/3; similarly Frot_bar
      - A = 6πη R_eq fT_bar, B = 8πη R_eq^3 Frot_bar
    Also returns Rg, Δ, S, p computed from G so the table has non-NaN shape metrics.
    """
    Roc = Roc_nm*1e-9
    bhel= (pitch_nm*1e-9)/(2.0*np.pi)
    G = gyration_tensor(coords)
    vals, Delta, S = shape_params_from_G(G)
    Rg = math.sqrt(float(np.sum(vals)))
    p_spheroid = p_from_S(S)

    # Semi-axes (vals sorted ascending: λ1≤λ2≤λ3 ⇒ a≤b≤c here; naming doesn't matter for averages)
    a, b, c = [math.sqrt(5.0*float(x)) for x in np.sort(vals)]
    Req = (a*b*c)**(1.0/3.0)
    def safe_sqrt(x): return math.sqrt(max(x, 1e-30))
    p_a = a / safe_sqrt(b*c)
    p_b = b / safe_sqrt(a*c)
    p_c = c / safe_sqrt(a*b)
    fTa, Fra = perrin_factors(p_a)
    fTb, Frb = perrin_factors(p_b)
    fTc, Frc = perrin_factors(p_c)
    fT_bar   = (fTa + fTb + fTc) / 3.0
    Frot_bar = (Fra + Frb + Frc) / 3.0
    A = 6.0*np.pi*eta*Req*fT_bar
    B = 8.0*np.pi*eta*(Req**3)*Frot_bar
    zeta = A + (B + A*(Roc**2)) / (bhel**2)
    D1   = kB*temp / zeta

    # expose descriptors unified with spheroid naming for the table (plus averaged fT/Frot)
    desc = dict(Rg=Rg, Delta=Delta, S=S, p=p_spheroid,
                Req=Req, fT_bar=fT_bar, Frot_bar=Frot_bar, A=A, B=B, zeta=zeta)
    return D1, desc

# --------------------------- helpers -------------------------
def nm(x): return x*1e9

def compute_budget(A, B, Roc_nm, pitch_nm):
    b = (pitch_nm*1e-9)/(2.0*np.pi)
    R_OC = Roc_nm*1e-9
    tA    = A
    tCurv = A*(R_OC**2)/(b**2)
    tRot  = B/(b**2)
    tot   = tA + tCurv + tRot
    if tot <= 0:
        return (float('nan'),)*3
    return (100.0*tA/tot, 100.0*tCurv/tot, 100.0*tRot/tot)

# --------------------------- CLI ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="BBXB batch runner (triaxial/spheroid); reports D1 scaled by 0.3."
    )
    ap.add_argument("--csv", required=True, help="CSV with columns: pdb,Roc_nm,pitch_nm")
    ap.add_argument("--eta", type=float, default=1.0e-3, help="Viscosity (Pa*s)")
    ap.add_argument("--temp", type=float, default=298.0, help="Temperature (K)")
    ap.add_argument("--assembly", action="store_true", help="Apply REMARK 350 BIOMT transforms")
    ap.add_argument("--useCA", action="store_true", help="Use only CA atoms for shape.")
    ap.add_argument("--chains", type=str, default=None, help="Comma-separated chain IDs to include.")
    ap.add_argument("--shape_model", choices=["triaxial","spheroid"], default="triaxial")
    ap.add_argument("--out", type=str, default=None, help="Optional output TSV with results.")
    ap.add_argument("--report_budget", action="store_true", help="Include budget % in console and table.")
    args = ap.parse_args()

    use_atoms = {"CA"} if args.useCA else None
    chain_filter = set(args.chains.split(",")) if args.chains else None

    # Labels for table columns (fT/Frot naming)
    if args.shape_model == "spheroid":
        fT_label, Frot_label = "fT", "Frot"
    else:
        fT_label, Frot_label = "fT_bar", "Frot_bar"

    # Load CSV
    rows = []
    with open(args.csv, newline='') as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)

    print("# Batch BBXB report\n")
    out_rows = []
    for r in rows:
        pdb_path = r["pdb"].strip()
        Roc_nm   = float(r["Roc_nm"]) if r.get("Roc_nm") else 2.5
        pitch_nm = float(r["pitch_nm"]) if r.get("pitch_nm") else 3.4

        try:
            mats = parse_biomt_from_pdb(pdb_path) if args.assembly else []
        except Exception:
            mats = []
        try:
            coords = load_pdb_coords(pdb_path, atom_names=use_atoms, chain_filter=chain_filter)
        except Exception as e:
            print(f"[ERROR] {pdb_path}: {e}", file=sys.stderr)
            continue
        if args.assembly and mats:
            coords = apply_biomt(coords, mats)

        if args.shape_model == "spheroid":
            D1_base, desc = bbxb_S_from_coords(coords, Roc_nm, pitch_nm, temp=args.temp, eta=args.eta)
            fT_val, Frot_val = desc["fT"], desc["Frot"]
        else:
            D1_base, desc = bbxb_T_from_coords(coords, Roc_nm, pitch_nm, temp=args.temp, eta=args.eta)
            fT_val, Frot_val = desc["fT_bar"], desc["Frot_bar"]

        # Shape metrics (available in both models now)
        Rg_nm      = nm(desc["Rg"])
        Delta_val  = desc["Delta"]
        S_val      = desc["S"]
        p_val      = desc["p"]
        Req_nm     = nm(desc["Req"])
        A, B, zeta = desc["A"], desc["B"], desc["zeta"]

        # Scale D1 by 0.3 (ruggedness proxy)
        D1_scaled = 0.3 * D1_base

        # Console block
        print(f"{Path(pdb_path).name}")
        print(f"R_OC={Roc_nm:.2f} nm; Pitch={pitch_nm:.2f} nm; T={args.temp:.1f} K; eta={args.eta:.2e} Pa*s")
        if args.shape_model == "spheroid":
            print(f"Rg={Rg_nm:.3f} nm;  \u0394={Delta_val:.4f};  S={S_val:.4f};  p={p_val:.3f}")
            print(f"Req={Req_nm:.3f} nm;  fT={fT_val:.4f};  Frot={Frot_val:.4f}")
        else:
            print(f"Rg={Rg_nm:.3f} nm;  \u0394={Delta_val:.4f};  S={S_val:.4f};  p={p_val:.3f}")
            print(f"Triaxial-Req={Req_nm:.3f} nm;  {fT_label}={fT_val:.4f};  {Frot_label}={Frot_val:.4f}")
        print(f"zeta_total = {zeta:.3e} N*s/m")
        print(f"D1 (scaled by 0.3) = {D1_scaled:.3e} m^2/s")
        if args.report_budget:
            a_pct, c_pct, r_pct = compute_budget(A, B, Roc_nm, pitch_nm)
            print(f"Budget: trans={a_pct:.1f}%, curv={c_pct:.1f}%, rot={r_pct:.1f}%")
        print("-"*60)

        a_pct, c_pct, r_pct = (float('nan'),)*3
        if args.report_budget:
            a_pct, c_pct, r_pct = compute_budget(A, B, Roc_nm, pitch_nm)

        out_rows.append([
            Path(pdb_path).name, f"{Roc_nm:.2f}", f"{pitch_nm:.2f}", f"{args.temp:.1f}", f"{args.eta:.3e}",
            f"{Rg_nm:.3f}", f"{Delta_val:.4f}", f"{S_val:.4f}", f"{p_val:.3f}",
            f"{Req_nm:.3f}", f"{fT_val:.4f}", f"{Frot_val:.4f}",
            f"{a_pct:.1f}", f"{c_pct:.1f}", f"{r_pct:.1f}",
            f"{zeta:.3e}", f"{D1_scaled:.3e}"
        ])

    # ---- Tabular summary ----
    header = [
        "pdb","Roc_nm","pitch_nm","T_K","eta_Pa_s",
        "Rg_nm","Delta","S","p","Req_nm", f"{fT_label}", f"{Frot_label}",
        "budget_trans_pct","budget_curv_pct","budget_rot_pct",
        "zeta_Ns_per_m","D1_BBXB_scaled_m2s"
    ]
    print("\n# Tabular summary")
    print("\t".join(header))
    for row in out_rows:
        print("\t".join(row))

    # Optional write (TSV)
    if args.out:
        with open(args.out, "w", newline="") as wf:
            w = csv.writer(wf, delimiter="\t")
            w.writerow(header)
            w.writerows(out_rows)
        print(f"\n[WROTE] {args.out}")

if __name__ == "__main__":
    main()
