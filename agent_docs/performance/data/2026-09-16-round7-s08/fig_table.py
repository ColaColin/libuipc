import re
def parse(path, want):
    out = {}
    sym = None
    for line in open(path):
        m = re.match(r'\s*Function (\S+?):', line)
        if m:
            sym = m.group(1)
        elif sym and re.match(r'\s+REG:', line):
            fm = re.search(r'REG:(\d+) STACK:(\d+)', line)
            cm = re.search(r'CONSTANT\[2\]:(\d+)', line)
            for fam, pat in want.items():
                if re.search(pat, sym):
                    ta = re.search(r'kernelILi(\d+)ELi(\d+)(?:ELi(\d+))?', sym)
                    tag = f"<{ta.group(1)},{ta.group(2)}" + (f",{ta.group(3)}" if ta.group(3) else "") + ">"
                    out[(fam, tag)] = (int(fm.group(1)), int(fm.group(2)), int(cm.group(1)) if cm else -1)
            sym = None
    return out
want = {
 'strain': r'StrainPlasticDiscreteShellBending_do_compute_gradient_hessian_kernel',
 'stress': r'StressPlasticDiscreteShellBending_do_compute_gradient_hessian_kernel',
 'dahl':   r'DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel',
 'hinge':  r'DiscreteShellBending_do_compute_gradient_hessian_kernel',
 'nhs2d':  r'NeoHookeanShell2D_do_compute_gradient_hessian_kernel',
}
main = parse('res_usage_main.txt', want)
branch = parse('res_usage_branch.txt', want)
print(f"{'family':8} {'inst':10} {'main (reg/stack/cmem2)':26} branch")
for key in sorted(set(main) | set(branch), key=str):
    m = main.get(key, ('NEW','',''))
    b = branch.get(key, ('GONE','',''))
    mark = '  =' if m == b else ('  +' if m == 'NEW' else ('  -' if b == 'GONE' else ' **CHANGED**'))
    print(f"{key[0]:8} {key[1]:10} {str(m):26} {str(b)}{mark}")
