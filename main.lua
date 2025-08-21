local M = {}

-- -----------------------------
-- Utility helpers
-- -----------------------------
local function assert_arg(cond, msg)
  if not cond then error(msg or "assertion failed") end
end

local function clone(t)
  local out = {}
  for i=1,#t do out[i] = t[i] end
  return out
end

-- -----------------------------
-- Vector operations
-- -----------------------------
function M.vec_add(a,b)
  assert(#a == #b, "vector size mismatch")
  local c = {}
  for i=1,#a do c[i] = a[i] + b[i] end
  return c
end

function M.vec_sub(a,b)
  assert(#a == #b, "vector size mismatch")
  local c = {}
  for i=1,#a do c[i] = a[i] - b[i] end
  return c
end

function M.vec_scale(a, s)
  local c = {}
  for i=1,#a do c[i] = a[i] * s end
  return c
end

function M.dot(a,b)
  assert(#a == #b, "vector size mismatch")
  local s = 0
  for i=1,#a do s = s + a[i]*b[i] end
  return s
end

function M.norm(a)
  return math.sqrt(M.dot(a,a))
end

-- -----------------------------
-- Matrix operations (row-major as table of rows)
-- -----------------------------
local function nrows(A) return #A end
local function ncols(A) return #A[1] end

function M.zeros(m,n)
  local A = {}
  for i=1,m do
    A[i] = {}
    for j=1,n do A[i][j] = 0 end
  end
  return A
end

function M.identity(n)
  local I = M.zeros(n,n)
  for i=1,n do I[i][i] = 1 end
  return I
end

function M.transpose(A)
  local m,n = nrows(A), ncols(A)
  local B = M.zeros(n,m)
  for i=1,m do for j=1,n do B[j][i] = A[i][j] end end
  return B
end

function M.mat_add(A,B)
  local m,n = nrows(A), ncols(A)
  assert(m==nrows(B) and n==ncols(B), "matrix size mismatch")
  local C = M.zeros(m,n)
  for i=1,m do for j=1,n do C[i][j] = A[i][j] + B[i][j] end end
  return C
end

function M.mat_mul(A,B)
  local m,p = nrows(A), ncols(A)
  assert(p == nrows(B), "inner dimensions must match")
  local n = ncols(B)
  local C = M.zeros(m,n)
  for i=1,m do
    for j=1,n do
      local s = 0
      for k=1,p do s = s + A[i][k]*B[k][j] end
      C[i][j] = s
    end
  end
  return C
end

function M.mat_vec_mul(A,x)
  local m,p = nrows(A), ncols(A)
  assert(#x == p)
  local y = {}
  for i=1,m do
    local s = 0
    for j=1,p do s = s + A[i][j]*x[j] end
    y[i] = s
  end
  return y
end

-- -----------------------------
-- Gaussian elimination (solve Ax=b) with partial pivoting
-- Returns x or nil+error
-- -----------------------------
function M.gauss_solve(A,b)
  local n = nrows(A)
  assert(n == ncols(A) and #b == n, "A must be square and match b")
  -- make copies
  local U = M.zeros(n,n)
  local bb = {}
  for i=1,n do
    for j=1,n do U[i][j] = A[i][j] end
    bb[i] = b[i]
  end

  -- elimination
  for k=1,n-1 do
    -- partial pivot
    local pivot_row = k
    local maxv = math.abs(U[k][k])
    for i=k+1,n do
      local v = math.abs(U[i][k])
      if v > maxv then maxv = v; pivot_row = i end
    end
    if maxv == 0 then return nil, "matrix is singular" end
    if pivot_row ~= k then U[k], U[pivot_row] = U[pivot_row], U[k]; bb[k], bb[pivot_row] = bb[pivot_row], bb[k] end
    for i=k+1,n do
      local factor = U[i][k] / U[k][k]
      U[i][k] = 0
      for j=k+1,n do U[i][j] = U[i][j] - factor * U[k][j] end
      bb[i] = bb[i] - factor * bb[k]
    end
  end
  if U[n][n] == 0 then return nil, "matrix is singular" end

  -- back substitution
  local x = {}
  for i=n,1,-1 do
    local s = bb[i]
    for j=i+1,n do s = s - U[i][j]*x[j] end
    x[i] = s / U[i][i]
  end
  return x
end

-- -----------------------------
-- LU decomposition with partial pivoting
-- Returns L, U, P (permutation vector)
-- -----------------------------
function M.lu_decompose(A)
  local n = nrows(A)
  assert(n == ncols(A), "A must be square")
  local LU = M.zeros(n,n)
  for i=1,n do for j=1,n do LU[i][j] = A[i][j] end end
  local P = {}
  for i=1,n do P[i] = i end

  for k=1,n do
    -- pivot
    local pivot_row = k
    local maxv = math.abs(LU[k][k])
    for i=k+1,n do local v = math.abs(LU[i][k]); if v > maxv then maxv = v; pivot_row = i end end
    if maxv == 0 then return nil, "singular matrix" end
    if pivot_row ~= k then LU[k], LU[pivot_row] = LU[pivot_row], LU[k]; P[k], P[pivot_row] = P[pivot_row], P[k] end
    -- elimination
    for i=k+1,n do
      LU[i][k] = LU[i][k] / LU[k][k]
      for j=k+1,n do LU[i][j] = LU[i][j] - LU[i][k] * LU[k][j] end
    end
  end
  -- extract L and U
  local L = M.identity(n)
  local U = M.zeros(n,n)
  for i=1,n do for j=1,n do
    if i > j then L[i][j] = LU[i][j] elseif i == j then L[i][j] = 1 elseif i < j then U[i][j] = LU[i][j] end
    if i<=j then U[i][j] = LU[i][j] end
  end end
  return L,U,P
end

function M.lu_solve(L,U,P,b)
  local n = #L
  -- apply permutation
  local bp = {}
  for i=1,n do bp[i] = b[P[i]] end
  -- forward
  local y = {}
  for i=1,n do
    local s = bp[i]
    for j=1,i-1 do s = s - L[i][j]*y[j] end
    y[i] = s
  end
  -- back
  local x = {}
  for i=n,1,-1 do
    local s = y[i]
    for j=i+1,n do s = s - U[i][j]*x[j] end
    x[i] = s / U[i][i]
  end
  return x
end

function M.det(A)
  local n = nrows(A)
  local _,U,P_or_err = M.lu_decompose(A)
  if not _ then -- LU returned err
    -- M.lu_decompose returns nil,err
    return nil, U
  else
    local L,U_mat,P = _,U,P_or_err
  end
  -- Actually we messed up return order above; simpler: redo here
  local L,U_mat,P2 = M.lu_decompose(A)
  if not L then return nil, U_mat end
  local det = 1
  for i=1,n do det = det * U_mat[i][i] end
  -- account for permutation sign
  local sign = 1
  for i=1,n do if P2[i] ~= i then sign = -sign; -- swap detection is approximate; better compute parity
  end end
  -- The above parity detection is wrong; compute parity properly
  -- compute permutation parity
  local visited = {}
  local parity = 1
  for i=1,n do
    if not visited[i] then
      local j = i
      local cycle_len = 0
      while not visited[j] do visited[j] = true; j = P2[j]; cycle_len = cycle_len + 1 end
      if cycle_len>0 and (cycle_len % 2 == 0) then parity = -parity end
    end
  end
  return det * parity
end

-- -----------------------------
-- Inverse via LU
-- -----------------------------
function M.inverse(A)
  local n = nrows(A)
  local L,U,P = M.lu_decompose(A)
  if not L then return nil, U end
  local inv = M.zeros(n,n)
  for j=1,n do
    local e = {}
    for i=1,n do e[i] = 0 end
    e[j] = 1
    local x = M.lu_solve(L,U,P,e)
    for i=1,n do inv[i][j] = x[i] end
  end
  return inv
end

-- -----------------------------
-- Power iteration for dominant eigenvalue/vector
-- -----------------------------
function M.power_iteration(A, opts)
  opts = opts or {}
  local maxit = opts.maxit or 1000
  local tol = opts.tol or 1e-10
  local n = nrows(A)
  local b = {}
  for i=1,n do b[i] = math.random() end
  local lambda = 0
  for it=1,maxit do
    local Ab = M.mat_vec_mul(A,b)
    local normb = M.norm(Ab)
    for i=1,n do b[i] = Ab[i] / normb end
    -- Rayleigh quotient
    local Ab2 = M.mat_vec_mul(A,b)
    local lambda_new = M.dot(b,Ab2)
    if math.abs(lambda_new - lambda) < tol then return lambda_new, b end
    lambda = lambda_new
  end
  return lambda, b
end

-- -----------------------------
-- Numerical calculus: derivatives, integrals, root finding, ODEs
-- -----------------------------

-- scalar derivative (central difference)
function M.derivative(f, x, h)
  h = h or 1e-6
  return (f(x+h) - f(x-h)) / (2*h)
end

-- gradient for multivariate function f(vector) returning numeric gradient
function M.gradient(f, x, h)
  h = h or 1e-6
  local n = #x
  local g = {}
  for i=1,n do
    local xp = clone(x); local xm = clone(x)
    xp[i] = xp[i] + h; xm[i] = xm[i] - h
    g[i] = (f(xp) - f(xm)) / (2*h)
  end
  return g
end

-- Newton-Raphson for scalar
function M.newton(f, df, x0, opts)
  opts = opts or {}
  local tol = opts.tol or 1e-10
  local maxit = opts.maxit or 100
  local x = x0
  for i=1,maxit do
    local fx = f(x)
    local dfx = df and df(x) or M.derivative(f,x)
    if dfx == 0 then return nil, "derivative zero" end
    local dx = fx/dfx
    x = x - dx
    if math.abs(dx) < tol then return x end
  end
  return nil, "no convergence"
end

-- Multivariate Newton using numeric jacobian
local function numeric_jacobian(F, x, h)
  h = h or 1e-6
  local n = #x
  local Fx = F(x)
  local m = #Fx
  local J = M.zeros(m,n)
  for j=1,n do
    local xp = clone(x); xp[j] = xp[j] + h
    local Fxp = F(xp)
    for i=1,m do J[i][j] = (Fxp[i] - Fx[i]) / h end
  end
  return J
end

function M.newton_multi(F, x0, opts)
  opts = opts or {}
  local tol = opts.tol or 1e-8
  local maxit = opts.maxit or 50
  local x = clone(x0)
  for it=1,maxit do
    local Fx = F(x)
    local err = M.norm(Fx)
    if err < tol then return x end
    local J = numeric_jacobian(F, x, opts.h)
    -- solve J * dx = Fx (we want dx to satisfy J dx = Fx, then x <- x - dx)
    local ok, msg
    local success
    local dx
    -- try LU
    local L,U,P = M.lu_decompose(J)
    if L then
      dx = M.lu_solve(L,U,P,Fx)
    else
      return nil, "Jacobian singular"
    end
    for i=1,#x do x[i] = x[i] - dx[i] end
  end
  return nil, "no convergence"
end

-- Simpson's rule (composite) for definite integral on [a,b]
function M.simpson(f, a, b, n)
  n = n or 100
  if n % 2 == 1 then n = n + 1 end
  local h = (b-a)/n
  local s = f(a) + f(b)
  for i=1,n-1 do
    local x = a + i*h
    s = s + (i%2==0 and 2 or 4) * f(x)
  end
  return s * h / 3
end

-- Adaptive Simpson's method
local function asr(f, a, b, eps, whole, fa, fb, fm)
  local m = (a+b)/2
  local h = b-a
  local f1 = f((a+m)/2)
  local f2 = f((m+b)/2)
  local left = (fa + 4*f1 + fm) * (h/4) / 6 * 6 -- simplified but we'll use standard formula below
  -- we'll compute Simpson for halves properly
  local Sleft = (fa + 4*f1 + fm) * (h/4) / 3
  local Sright = (fm + 4*f2 + fb) * (h/4) / 3
  local S2 = Sleft + Sright
  if math.abs(S2 - whole) <= 15*eps then return S2 + (S2 - whole)/15 end
  return asr(f,a,m,eps/2,Sleft,fa,fm,f1) + asr(f,m,b,eps/2,Sright,fm,fb,f2)
end

function M.adaptive_simpson(f, a, b, eps)
  eps = eps or 1e-8
  local m = (a+b)/2
  local fa, fb, fm = f(a), f(b), f(m)
  local whole = (fa + 4*fm + fb) * (b-a) / 6
  return asr(f,a,b,eps,whole,fa,fb,fm)
end

-- Runge-Kutta 4 for system dx/dt = f(t,x), x vector
function M.rk4_step(f, t, x, h)
  local k1 = f(t, x)
  local x2 = M.vec_add(x, M.vec_scale(k1, h/2))
  local k2 = f(t + h/2, x2)
  local x3 = M.vec_add(x, M.vec_scale(k2, h/2))
  local k3 = f(t + h/2, x3)
  local x4 = M.vec_add(x, M.vec_scale(k3, h))
  local k4 = f(t + h, x4)
  local out = {}
  for i=1,#x do out[i] = x[i] + h*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6 end
  return out
end

function M.rk4_solve(f, t0, x0, h, steps)
  local t = t0
  local x = clone(x0)
  for i=1,steps do
    x = M.rk4_step(f, t, x, h)
    t = t + h
  end
  return x
end
