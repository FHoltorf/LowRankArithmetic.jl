using ITensors
import Base.+, Base.-

struct myLRTucker
    core
    leaves
end

function shift(T::myLRTucker, ν)
    leaves = deepcopy(T.leaves)
    for idx in keys(ν)
        basis = matrix(leaves[idx]) 
        if ν[idx] < 0 
            basis[1:end+ν[idx],:] .= basis[1-ν[idx]:end, :]
            basis[end+ν[idx]+1:end, :] .= 0 
        elseif ν[idx] > 0
            basis[1+ν[idx]:end, :] .= basis[1:end-ν[idx], :]
            basis[1:ν[idx], :] .= 0
        end
    end
    return myLRTucker(T.core,leaves)
end

function combine(Atensor, Btensor)
    A = matrix(Atensor)
    B = matrix(Btensor)
    rA, rB = size(A,2), size(B,2)
    r_new = rA*rB
    U = ones(size(A,1), r_new)
    Acols = [@view A[:,i] for i in 1:rA]
    Bcols = [@view B[:,i] for i in 1:rB]
    k = 0
    for r1 in 1:rA, r2 in 1:rB
        k += 1
        U[:, k] = Acols[r1] .* Bcols[r2]
    end
    return U, k
end

function hadamard(T::myLRTucker, Q::myLRTucker)
    T_bases = keys(T.leaves)
    R_leaves = Dict()
    ranks = Dict()
    idcs = Dict()
    for base in T_bases
        new_base, new_rank = combine(T.leaves[base], Q.leaves[base])
        new_idx = Index(new_rank)
        idcs[base] = new_idx
        ranks[base] = new_rank
        R_leaves[base] = ITensor(new_base, base, new_idx)
    end
    R_core = ITensor(collect(values(idcs))...)
    Q_ranks = size(Q.core)
    for idx_T in eachindex(T.core), idx_Q in eachindex(Q.core)
        new_lin_idx = CartesianIndex(Q_ranks .* (idx_T.I .- 1) .+ idx_Q.I)
        R_core[new_lin_idx] = T.core[idx_T]*Q.core[idx_Q]
    end
    return myLRTucker(R_core, R_leaves)
end

function +(T::myLRTucker, Q::myLRTucker)
    T_bases = keys(T.leaves)
    R_leaves = Dict()
    ranks = Dict()
    idcs = Dict()
    for base in T_bases
        r1, r2 = size(T.leaves[base],2), size(Q.leaves[base],2)
        new_idx = Index(r1+r2)
        idcs[base] = new_idx
        ranks[base] = r1+r2
        UT = matrix(T.leaves[base])
        UQ = matrix(Q.leaves[base])
        R_leaves[base] = ITensor(hcat(UT,UQ), base, new_idx)
    end
    Tranks = size(T.core)
    Qranks = size(Q.core)
    R_core = ITensor(zeros(Tranks .+ Qranks), collect(values(idcs))...)
    Tblock = [1:r for r in size(T.core)]
    Qblock = [Tranks[i]+1:Tranks[i]+Qranks[i] for i in eachindex(Qranks)]
    R_core[Tblock...] = deepcopy(ITensors.tensor(T.core))
    R_core[Qblock...] = deepcopy(ITensors.tensor(Q.core))
    return myLRTucker(R_core, R_leaves)
end

function -(T::myLRTucker)
    return myLRTucker(-T.core, T.leaves)
end

function tensor(A::myLRTucker)
    return A.core * prod(A.leaves[key] for key in keys(A.leaves))
end

function cme(P, As, ν)
    return sum(shift(hadamard(As[r],P), ν[r]) for r in eachindex(As)) + sum(hadamard(-A,P) for A in As)
end

function simple_cme!(dP, P, As, ν)
    nA, nB, nC = size(P)
    s = Int[0,0,0]
    for iA in 1:nA, iB in 1:nB, iC in 1:nC
        dP[iA,iB,iC] = 0
        for r in eachindex(As)
            s[1] = iA-ν[r][1]
            s[2] = iB-ν[r][2]
            s[3] = iC-ν[r][3]
            flag = true
            if s[1] <= 0 || s[1] > nA 
                flag = false
            elseif s[2] <= 0 || s[2] > nB
                flag = false
            elseif s[3] <= 0 || s[3] > nC
                flag = false
            end
            if flag
                dP[iA,iB,iC] += As[r][s[1],s[2],s[3]] * P[s[1],s[2],s[3]] - As[r][iA,iB,iC] * P[iA,iB,iC]
            end
        end
    end 
end


# A + B <-> C 
# ∅ -> C -> ∅
function generate_data(nA, nB, nC)
    k = [1.0, 10.0, 0.4, 12.0]

    A = Index(length(nA))
    B = Index(length(nB))
    C = Index(length(nC))

    rA = Index(1)
    rB = Index(1)
    rC = Index(1)
    K = [ITensor(rate_coeff,rA, rB, rC) for rate_coeff in k]
    ν = [Dict(A => -1, B => -1, C => 1), 
        Dict(A => 1, B => 1, C => -1), 
        Dict(C => 1, A => 0, B=> 0),
        Dict(C => -1, A => 0, B=> 0)]
    Arange = ITensor(reshape(collect(nA),length(nA),1), A, rA)
    Brange = ITensor(reshape(collect(nB),length(nB),1), B, rB)
    Crange = ITensor(reshape(collect(nC),length(nC),1), C, rC)
    Aind_range = ITensor(ones(length(nA),1), A, rA)
    Bind_range = ITensor(ones(length(nB),1), B, rB)
    Cind_range = ITensor(ones(length(nC),1), C, rC)

    # A + B -> C
    A1 = myLRTucker(K[1], Dict(A => Arange, B => Brange, C => Cind_range))
    # C -> A + B
    A2 = myLRTucker(K[2], Dict(A => Aind_range, B => Bind_range, C => Crange))
    # ∅ -> C
    A3 = myLRTucker(K[3], Dict(A => Aind_range, B => Bind_range, C => Cind_range))
    # C -> ∅
    A4 = myLRTucker(K[4], Dict(A => Aind_range, B => Bind_range, C => Crange))

    As = [A1,A2,A3,A4]
    As_full = [tensor(A) for A in As] 
    return A, B, C, As, As_full, ν
end

ranks = [1,2,5,10]
ns = [50, 100, 150, 200, 300, 500, 700]
lr_times = Dict()
function sample_cme(P_lr, As, ν, samples)
    for s in 1:samples
        test = cme(P_lr, As, ν)
    end
    return nothing
end

for n in ns
    A,B,C,As,As_full,ν = generate_data(0:n,0:n,0:n)
    for rank in ranks
        core = randomITensor(Index(rank),Index(rank),Index(rank))
        idcs = inds(core)
        leaves = Dict(A => randomITensor(A,idcs[1]), B => randomITensor(B,idcs[2]), C => randomITensor(C,idcs[3]))
        P_lr = myLRTucker(core, leaves)
        t_lr = @benchmark cme($P_lr, $As, $ν)
        #@elapsed sample_cme(P_lr, As, ν, samples)
        lr_times[n, rank] = t_lr#/samples
    end
end

full_times = Dict()
samples = 10
for n in ns
    A,B,C,As,As_full,ν = generate_data(0:n,0:n,0:n)
    P_full = randomITensor(A,B,C)
    dP_full = similar(P_full)
    ν_full = [[ν[r][A], ν[r][B], ν[r][C]] for r in eachindex(ν)]
    simple_cme!(dP_full, P_full, As_full, ν_full)
    t_full = []
    for s in 1:samples
        push!(t_full, @elapsed simple_cme!(dP_full, P_full, As_full, ν_full))
    end
    full_times[n] = t_full
end

using CairoMakie
fig = Figure(fontsize=18)
ax = Axis(fig[1,1], xlabel = "n", ylabel = "CPU time [s]", yscale=log10)
for r in ranks
    scatterlines!(ax, ns, [mean(lr_times[n,r].times)*1e-6 for n in ns], color = :dodgerblue, label = "rank $r")
end
scatterlines!(ax, ns, [mean(full_times[n]) for n in ns], color=:black)
display(fig)
save(string(@__DIR__,"/tucker_lra.pdf"), fig)

#P_full = tensor(P_lr)
#dP_full = similar(P_full)
#ν_full = [[ν[r][A], ν[r][B], ν[r][C]] for r in eachindex(ν)]
# @time simple_cme!(dP_full, P_full, As_full, ν_full)