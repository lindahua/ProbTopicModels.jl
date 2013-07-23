# Latent Dirichlet Allocation

## LDA model

immutable LDAModel
	dird::Dirichlet
	topics::Matrix{Float64}  # V x K
	tlogp::Matrix{Float64}   # K x V

	function LDAModel(alpha::Vector{Float64}, topics::Matrix{Float64}) 
		@check_argdims length(alpha) == size(topics, 2)
		new(Dirichlet(alpha), topics, log(topics'))
	end

	function LDAModel(alpha::Float64, topics::Matrix{Float64}) 
		K = size(topics, 2)
		new(Dirichlet(K, alpha), topics, log(topics'))
	end
end

nterms(m::LDAModel) = size(m.topics, 1)
ntopics(m::LDAModel) = size(m.topics, 2)

# sampling

function randdoc(model::LDAModel, theta::Vector{Float64}, len::Int)
	@check_argdims length(theta) == ntopics(model)

	p = model.topics * theta
	wcounts = rand(Multinomial(len, p))
	ts = find(wcounts)
	SDocument(ts, wcounts[ts])
end

randdoc(model::LDAModel, len::Int) = randdoc(s, rand(s.tp_distr), len)


#####################################################################
#
#   Variational Inference based on LDA
#
#   Inputs:
#	- LDA model (K topics)
#	- document (n distinct words)
#
#	Outputs:
#	- vgam:         Variational gamma (K,)
#	- elogtheta:    E[log(theta)] based on the variational
#	                Dirichlet distribution (K,)
#	- vphi:         per-word topic assignment (K,n)
#   - tocweights:   overall topic weights (vphi * counts)
#
#####################################################################

immutable LDAVarInfer
	maxiter::Int
	tol::Float64
	verbose::Int

	function LDAVarInfer(;maxiter::Integer=100, tol::Real=1.0e-4, display::Symbol=:none)
		new(int(maxiter), float64(tol), verbosity_level(display))
	end
end

immutable LDAVarInferResults
	gamma::Vector{Float64}
	elogtheta::Vector{Float64}
	phi::Matrix{Float64}
	tocweights::Vector{Float64}
end

mean_theta(r::LDAVarInferResults) = r.gamma * inv(sum(r.gamma))

function _dirichlet_entropy(a::Vector{Float64}, elog::Vector{Float64})
	K = length(a)
	s = 0.
	ent = 0.
	for k in 1 : K
		ak = a[k]
		s += ak
		ent += (lgamma(ak) - (ak - 1.0) * elog[k])
	end
	ent -= lgamma(s)
end


function lda_varinfer_objv(model::LDAModel, doc::SDocument, 
	vgam::Vector{Float64}, elogtheta::Vector{Float64}, vphi::Matrix{Float64}, tocweights::Vector{Float64})

	K = ntopics(model)
	alpha::Vector{Float64} = model.dird.alpha
	tlogp = model.tlogp
	terms = doc.terms
	h = doc.counts

	t_lptheta = 0.
	for k in 1 : K
		t_lptheta += (alpha[k] - 1.) * elogtheta[k]
	end

	t_lptoc = dot(tocweights, elogtheta)

	gam_ent = _dirichlet_entropy(vgam, elogtheta)

	t_lpword = 0.
	t_phi_ent = 0.

	for i in 1 : length(terms)
		w = terms[i]
		lpw_i = 0.
		pent_i = 0.
		for k in 1 : K
			pv = vphi[k,i]
			if pv > 0.
				lpw_i += pv * tlogp[k,w]
				pent_i -= pv * log(pv)
			end
		end
		t_lpword += h[i] * lpw_i
		t_phi_ent += h[i] * pent_i
	end

	t_lptheta + t_lptoc + t_lpword + gam_ent + t_phi_ent
end


function lda_varinfer_init!(model::LDAModel, doc::SDocument, 
	vgam::Vector{Float64}, elogtheta::Vector{Float64}, vphi::Matrix{Float64}, tocweights::Vector{Float64})

	K::Int = ntopics(model)
	n::Int = length(doc.terms)
	alpha::Vector{Float64} = model.dird.alpha
	alpha0::Float64 = model.dird.alpha0

	@check_argdims length(vgam) == K
	@check_argdims size(vphi, 1) == K && size(vphi, 2) == n

	avg_tocweight = doc.sum_counts / K
	digam_sum = digamma(alpha0 + doc.sum_counts)

	for k in 1 : K
		vgam[k] = alpha[k] + avg_tocweight
		elogtheta[k] = digamma(vgam[k]) - digam_sum
		tocweights[k] = avg_tocweight
	end

	fill!(vphi, 1.0 / K)
end


function lda_varinfer_update!(model::LDAModel, doc::SDocument, 
	γ::Vector{Float64}, elogθ::Vector{Float64}, φ::Matrix{Float64}, τ::Vector{Float64}, 
	method::LDAVarInfer)

	# Notations:
	# γ:      variational Dirichlet parameter of topic proportions
	# elogθ:  expectation of log(θ) w.r.t. Dirichlet(γ)
	# φ:      per-word soft-assignment of topics, K x n matrix
	# τ:      topic weights, i.e. sum(φ * h, 2)

	terms = doc.terms
	h = doc.counts
	α::Vector{Float64} = model.dird.alpha
	α0::Float64 = model.dird.alpha0
	digam_sum::Float64 = digamma(α0 + doc.sum_counts)
	tlogp = model.tlogp

	maxiter = method.maxiter
	tol = method.tol
	verbose = method.verbose
	K::Int = ntopics(model)
	n::Int = length(terms)

	@check_argdims length(γ) == K
	@check_argdims size(φ, 1) == K && size(φ, 2) >= n

	# prepare for the updating loop

	converged = false
	it = 0
	objv = lda_varinfer_objv(model, doc, γ, elogθ, φ, τ)

	if verbose >= VERBOSE_PROC
		@printf("%6s    %12s     %12s\n", "iter", "objective", "obj.change")
		println("------------------------------------------")
		@printf("%6d    %12.4e\n", 0, objv)
	end

	# updating loop

	while !converged && it < maxiter
		it += 1

		# reset τ
		fill!(τ, 0.)

		# per-word topic assignment (soft) --> φ, a K x n matrix

		o::Int = 0  # base offset for φ, φ[o + k] -- φ[k, i]
		for i in 1 : n
			@inbounds w = terms[i]

			# evidences in log-scale
			mp = -Inf			
			for k in 1 : K
				φ[o + k] = v = tlogp[k,w] + elogθ[k]
				if v > mp
					mp = v
				end
			end

			# softmax (normalize φ)
			# accumulate to γ at the same time

			s = 0.
			for k in 1 : K
				ki = o + k
				@inbounds s += (φ[ki] = exp(φ[ki] - mp))				
			end

			inv_s = 1.0 / s
			for k in 1 : K
				@inbounds pk = (φ[o + k] *= inv_s)
				@inbounds γ[k] = α[k] + (τ[k] += pk * h[i])
			end

			o += K
		end

		# update elogθ based on new γ
		for k in 1 : K
			@inbounds elogθ[k] = digamma(γ[k]) - digam_sum
		end

		# decide convergence

		objv_pre = objv
		objv = lda_varinfer_objv(model, doc, γ, elogθ, φ, τ)
		converged = abs(objv - objv_pre) < tol

		if verbose >= VERBOSE_ITER
			@printf("%6d    %12.4e     %12.4e\n", it, objv, objv - objv_pre)
		end
	end

	if verbose >= VERBOSE_PROC
		if converged
			println("Inference converged with $it iterations.")
		else
			println("Inference terminated after $it iterations without convergence.")
		end
	end
end


function infer(model::LDAModel, doc::SDocument, method::LDAVarInfer)
	n = length(doc.terms)
	K = ntopics(model)

	γ = Array(Float64, K)
	elogθ = Array(Float64, K)
	φ = Array(Float64, K, n)
	τ = Array(Float64, K)

	lda_varinfer_init!(model, doc, γ, elogθ, φ, τ)
	lda_varinfer_update!(model, doc, γ, elogθ, φ, τ, method)
	return LDAVarInferResults(γ, elogθ, φ, τ)
end




