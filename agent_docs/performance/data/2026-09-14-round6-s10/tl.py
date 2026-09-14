import csv,sys
p=sys.argv[1]; key=sys.argv[2] if len(sys.argv)>2 else 'do_assemble_kernel<(bool)0, (int)1'
a=int(sys.argv[3]); b=int(sys.argv[4])
rows=[]
with open(p) as f:
    for d in csv.DictReader(f):
        if not d['Start (ns)']: continue
        rows.append((int(d['Start (ns)']),int(d['Duration (ns)']),d['Strm'],d['GrdX'],d['BlkX'],d['Reg/Trd'],d['Name']))
rows.sort()
def short(n):
    n=n.split('(')[0]
    for pre in ('void uipc::backend::cuda::<unnamed>::','uipc::backend::cuda::<unnamed>::','uipc::backend::cuda::','void uipc::backend::cuda_tool::','uipc::backend::cuda_tool::'):
        n=n.replace(pre,'')
    return n[:60]
ix=[i for i,r in enumerate(rows) if key in r[6]]
c=ix[len(ix)//2]; s0=rows[c][0]
for i in range(c+a,c+b):
    s,d,st,g,bl,reg,n=rows[i]
    print(f"{(s-s0)/1000.0:9.1f} {d/1000.0:8.1f} s{st:>3} g{g:>5} b{bl:>5} r{reg:>4} {short(n)}")
