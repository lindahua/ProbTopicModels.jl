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

immutable LDAVarInferOptions
	maxiter::Int
	tol::Float64
	verbose::Int

	function LDAVarInferOptions(maxiter::Int, tol::Float64, disp::Symbol)
		new(maxiter, tol, verbosity_level(disp))
	end
end

immutable LDAVarInferResults
	vgam::Vector{Float64}
	elogtheta::Vector{Float64}
	vphi::Matrix{Float64}
	tocweights::Vector{Float64}
end


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
	vgam::Vector{Float64}, elogtheta::Vector{Float64}, vphi::Matrix{Float64}, tocweights::Vector{Float64}, 
	options::LDAVarInferOptions)

	terms = doc.terms
	h = doc.counts
	alpha::Vector{Float64} = model.dird.alpha
	alpha0::Float64 = model.dird.alpha0
	digam_sum::Float64 = digamma(alpha0 + doc.sum_counts)
	tlogp = model.tlogp

	maxiter = options.maxiter
	tol = options.tol
	verbose = options.verbose
	K::Int = ntopics(model)
	n::Int = length(terms)

	@check_argdims length(vgam) == K
	@check_argdims size(vphi, 1) == K && size(vphi, 2) == n

	# prepare for the updating loop

	converged = false
	it = 0
	objv = lda_varinfer_objv(model, doc, vgam, elogtheta, vphi, tocweights)

	if verbose >= VERBOSE_PROC
		@printf("%6s    %12s     %12s\n", "iter", "objective", "obj.change")
		println("------------------------------------------")
		@printf("%6d    %12.4e\n", 0, objv)
	end

	# updating loop

	while !converged && it < maxiter
		it += 1

		# reset tocweights
		fill!(tocweights, 0.)

		# per-word topic assignment (soft)
		for i in 1 : n
			w = terms[i]

			# compute in log-scale
			mp = -Inf			
			for k in 1 : K
				vphi[k,i] = v = tlogp[k,w] + elogtheta[k]
				if v > mp
					mp = v
				end
			end

			# softmax (normalize vphi)
			# accumulate to vgam at the same time

			s = 0.
			for k in 1 : K
				s += (vphi[k,i] = exp(vphi[k,i] - mp))				
			end

			inv_s = 1.0 / s
			for k in 1 : K
				pk = (vphi[k,i] *= inv_s)
				vgam[k] = alpha[k] + (tocweights[k] += pk * h[i])
			end
		end

		# update elogtheta based on new vgam
		for k in 1 : K
			elogtheta[k] = digamma(vgam[k]) - digam_sum
		end

		# decide convergence

		objv_pre = objv
		objv = lda_varinfer_objv(model, doc, vgam, elogtheta, vphi, tocweights)
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


function lda_varinfer(model::LDAModel, doc::SDocument, options::LDAVarInferOptions)
	n = length(doc.terms)
	K = ntopics(model)

	vgam = Array(Float64, K)
	elogtheta = Array(Float64, K)
	vphi = Array(Float64, K, n)
	tocweights = Array(Float64, K)

	lda_varinfer_init!(model, doc, vgam, elogtheta, vphi, tocweights)
	lda_varinfer_update!(model, doc, vgam, elogtheta, vphi, tocweights, options)
	return LDAVarInferResults(vgam, elogtheta, vphi, tocweights)
end

function lda_varinfer(model::LDAModel, doc::SDocument; maxiter::Int=100, tol::Float64=1.0e-4, display=:iter)
	lda_varinfer(model, doc, LDAVarInferOptions(maxiter, tol, display))
end

