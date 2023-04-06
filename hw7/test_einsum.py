import torch


x = torch.rand(4 , 4 )
y0 = torch.rand( 5 )
y1 = torch.rand( 4 )
z0 = torch.rand (3 , 2 , 5 )
z1 = torch.rand (3 , 5 , 4 )
w = torch.rand (2 , 3 , 4 , 5 )
r0 = torch.rand (2 , 5 )
r3 = torch.rand (2 , 5 )
r1 = torch.rand (3 , 5 , 4 )
r2 = torch.rand (2 , 4 )
s0 = torch.rand (2 , 3 , 5 , 7 )
s1 = torch.rand ( 11 , 3 , 17 , 5 )

a0 = torch . einsum ( 'i', y0 )
a1 = torch . einsum ( 'ij', x )
a2 = torch . einsum ( 'ijk' , z0 )


out = y0
assert((out==a0).sum())
out = x
assert((out==a1).sum())
out = z0
assert((out==a2).sum())

# permute
b0 = torch . einsum ( 'ij -> ji ', x )
b1 = torch . einsum ( 'ba' , x )
b2 = torch . einsum ( 'jki', z0 ) # jki -> ijk: 2, 0, 1
b3 = torch . einsum ( 'ijk -> kij' , z0 )
b4 = torch . einsum ( 'kjil', w ) # kjil -> ijkl: 2, 1, 0, 3
b5 = torch . einsum ( '...ij -> ...ji', w )
b6 = torch . einsum ( 'abc... -> cba...', w )

out = torch.permute(x, (1,0))
assert((out==b0).sum())
out = torch.permute(x, (1,0))
assert((out==b1).sum())
out = torch.permute(z0, (2,0,1))
assert((out==b2).sum())
out = torch.permute(z0, (2,0,1))
assert((out==b3).sum())
out = torch.permute(w, (2,1,0,3))
assert((out==b4).sum())
out = torch.permute(w, (0,1,3,2))
assert((out==b5).sum())
out = torch.permute(w, (2,1,0,3))
assert((out==b6).sum())

# trace
c = torch.einsum('ii', x)
out = torch.trace(x)
assert((torch.abs(out-c)).sum()<1e-5)

# sum
d0 = torch.einsum('ij->', x)
d1 = torch.einsum('xyz->', z0)
d2 = torch.einsum('ijkl->', w)

out = torch.sum(x)
assert((torch.abs(out-d0)).sum()<1e-5)
out = torch.sum(z0)
assert((torch.abs(out-d1)).sum()<1e-5)
out = torch.sum(w)
assert((torch.abs(out-d2)).sum()<1e-5)

# sum axis
e0 = torch.einsum('ijk->i', z0)
e1 = torch.einsum('ijk->j', z0)
e2 = torch.einsum('ijk->ij', z0)

out = torch.sum(z0, dim=(1,2))
assert((torch.abs(out-e0)).sum()<1e-5)
out = torch.sum(z0, dim=(0,2))
assert((torch.abs(out-e1)).sum()<1e-5)
out = torch.sum(z0, dim=2)
assert((torch.abs(out-e2)).sum()<1e-5)

 # matrix-vector
f0 = torch.einsum('ij,j->i', r0, y0)
f1 = torch.einsum('i,jki->jk', y1, r1)

out = torch.matmul(r0, y0)
assert((torch.abs(out-f0)).sum()<1e-5)
out = torch.matmul(r1, y1)
assert((torch.abs(out-f1)).sum()<1e-5)

# vector-vector outer product
g0 = torch.einsum('i,j->ij', y0, y1)
g1 = torch.einsum('a,b,c,d->abcd', y0, y1, y0, y1)

out = y0[:,None] * y1
assert((torch.abs(out-g0)).sum()<1e-5)
out = ((y0[:,None] * y1)[: , :, None] * y0)[:, :, :, None] * y1
assert((torch.abs(out-g1)).sum()<1e-5)

# batch mm
h0 = torch.einsum('bij,bjk->bik', z0, z1)
h1 = torch.einsum('bjk,bij->bik', z1, z0)

out = torch.matmul(z0, z1)
assert((torch.abs(out-h0)).sum()<1e-5)
out = torch.matmul(z0, z1)
assert((torch.abs(out-h1)).sum()<1e-5)

# bilinear
i = torch.einsum('bn,anm,bm->ba', r0, r1, r2)

m = torch.nn.Bilinear(5, 4, 3, bias=False)
m.weight.data = r1
out = m(r0, r2)
assert((torch.abs(out-i)).sum()<1e-5)

# tensor contraction
j = torch.einsum('pqrs,tqvr->pstv', s0, s1)

h = torch.einsum('ab, ab -> b', r0, r3)
print(r0)
print(r3)
print(h)
print(torch.sum(r0 * r3, dim=0))

