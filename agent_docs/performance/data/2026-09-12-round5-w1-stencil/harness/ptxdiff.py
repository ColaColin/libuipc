import re,sys,difflib
t=open(sys.argv[1]).read()
names=[n.rstrip("(") for n in re.findall(r"\.entry (\S+)", t)]
pl=[n for n in names if "_kernelILb1ELb1ELb1E" in n][0]
occ=[n for n in names if "_kernel_occILb1ELb1ELb1E" in n][0]
def body(name):
    i=t.index(".entry "+name)
    j=t.find(".entry ", i+10)
    if j<0: j=len(t)
    b=t[i:j]
    b=re.sub(r"\.maxntid[^\n]*\n","",b)
    b=re.sub(r"\.minnctapersm[^\n]*\n","",b)
    b=re.sub(r"__nv_static_\w*gradient_hessian\w*","KERNEL",b)
    b=re.sub(r"\$L__BB\d+_","$L__BB_",b)
    b=re.sub(r"callseq \d+","callseq N",b)
    b=re.sub(r"__local_depot\d+","__local_depot",b)
    return b
a=body(pl); b=body(occ)
print("plain bytes",len(a),"occ bytes",len(b))
if a==b:
    print("PTX BODIES IDENTICAL -- the two kernels differ only by .maxntid/.minnctapersm")
else:
    d=[l for l in difflib.unified_diff(a.splitlines(),b.splitlines(),lineterm="",n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
    print("PTX BODIES DIFFER in",len(d),"lines:")
    print("\n".join(x[:160] for x in d[:30]))
