# Latent Dirichlet Allocation

## LDA model

immutable LDAModel
	dird::Dirichlet
	topics::Matrix{Float64}  # V x K
	tlogp::Matrix{Float64}   # K x V

	function LDAModel(dird::Dirichlet, topics::Matrix{Float64})
		@check_argdims dim(dird) == size(topics,2)
		new(dird, topics, log(topics'))
	end

	LDAModel(alpha::Vector{Float64}, topics::Matrix{Float64}) = LDAModel(Dirichlet(alpha), topics)
	LDAModel(alpha::Float64, topics::Matrix{Float64}) = LDAModel(Dirichlet(size(topics,2),alpha), topics)
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

	function LDAVarInferResults(K::Int, nmax::Int)
		new(Array(Float64, K), 
			Array(Float64, K), 
			Array(Float64, K, nmax), 
			Array(Float64, K))
	end
end

ntopics(r::LDAVarInferResults) = length(r.gamma)

mean_theta(r::LDAVarInferResults) = r.gamma * inv(sum(r.gamma))

function _dirichlet_entropy(α::Vector{Float64}, elogθ::Vector{Float64})
	K = length(α)
	s = 0.
	ent = 0.
	for k in 1 : K
		@inbounds αk = α[k]
		s += αk
		@inbounds ent += (lgamma(αk) - (αk - 1.0) * elogθ[k])
	end
	ent -= lgamma(s)
end


function objective(model::LDAModel, doc::SDocument, r::LDAVarInferResults)
	# compute the objective of LDA variational inference

	K = ntopics(model)
	α::Vector{Float64} = model.dird.alpha
	tlogp = model.tlogp
	terms = doc.terms
	h = doc.counts

	γ = r.gamma
	elogθ = r.elogtheta
	φ = r.phi
	τ = r.tocweights

	t_lpθ = 0.
	for k in 1 : K
		t_lpθ += (α[k] - 1.) * elogθ[k]
	end

	t_lptoc = dot(τ, elogθ)

	γent = _dirichlet_entropy(γ, elogθ)

	t_lpword = 0.
	t_φent = 0.

	for i in 1 : length(terms)
		w = terms[i]
		lpw_i = 0.
		pent_i = 0.
		for k in 1 : K
			pv = φ[k,i]
			if pv > 0.
				@inbounds lpw_i += pv * tlogp[k,w]
				pent_i -= pv * log(pv)
			end
		end
		t_lpword += h[i] * lpw_i
		t_φent += h[i] * pent_i
	end

	t_lpθ + t_lptoc + t_lpword + γent + t_φent
end


function update_per_gamma!(model::LDAModel, doc::SDocument, r::LDAVarInferResults)
	# update the result struct based on the gamma field

	K::Int = ntopics(model)
	@assert K == ntopics(r)

	# get model & doc fields
	tlogp = model.tlogp
	terms = doc.terms
	h = doc.counts
	
	# fields of r
	γ = r.gamma
	elogθ = r.elogtheta
	φ = r.phi
	τ = r.tocweights

	γ0 = sum(γ)
	dγ0 = digamma(γ0)

	for k = 1:K
		@inbounds elogθ[k] = digamma(γ[k]) - dγ0
	end

	soft_topic_assign!(tlogp, elogθ, terms, h, φ, τ)
	r
end

function lda_varinfer_init!(model::LDAModel, doc::SDocument, r::LDAVarInferResults)
	# Inplace initialization of LDA variational inference results

	K::Int = ntopics(model)
	α::Vector{Float64} = model.dird.alpha

	avg_tocweight = doc.sum_counts / K
	γ = r.gamma
	for k = 1:K
		γ[k] = α[k] + avg_tocweight
	end

	update_per_gamma!(model, doc, r)
end


function lda_varinfer_init(model::LDAModel, doc::SDocument)
	# Create an initialized variational inference result struct

	K = ntopics(model)
	lda_varinfer_init!(model, doc, LDAVarInferResults(K, histlength(doc)))
end


function lda_varinfer_update!(model::LDAModel, doc::SDocument, r::LDAVarInferResults, method::LDAVarInfer)
	# core function of LDA variational inference

	# options

	maxiter = method.maxiter
	tol = method.tol
	verbose = method.verbose

	# basic variables

	K::Int = ntopics(model)
	@check_argdims ntopics(r) == K
	γ::Vector{Float64} = r.gamma
	τ::Vector{Float64} = r.tocweights
	α::Vector{Float64} = model.dird.alpha

	# prepare for the updating loop

	converged = false
	it = 0
	objv = objective(model, doc, r)

	if verbose >= VERBOSE_PROC
		@printf("%6s    %12s     %12s\n", "iter", "objective", "obj.change")
		println("------------------------------------------")
		@printf("%6d    %12.4e\n", 0, objv)
	end

	# updating loop

	while !converged && it < maxiter
		it += 1

		# update γ
		for k = 1:K
			@inbounds γ[k] = α[k] + τ[k]
		end

		# update others
		update_per_gamma!(model, doc, r)

		# decide convergence

		objv_pre = objv
		objv = objective(model, doc, r)
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

	return objv
end


function infer(model::LDAModel, doc::SDocument, method::LDAVarInfer)
	n = length(doc.terms)
	K = ntopics(model)

	r::LDAVarInferResults = lda_varinfer_init(model, doc)
	objv = lda_varinfer_update!(model, doc, r, method)
	return (r, objv)
end


#####################################################################
#
#   Variational EM for LDA learning
#
#####################################################################

immutable LDAVarLearn
	maxiter::Int
	tol::Float64
	vinfer_iter::Int
	vinfer_tol::Float64
	fix_topics::Bool
	fix_alpha::Bool
	verbose::Int

	function LDAVarLearn(;
		maxiter::Integer=200,
		tol::Float64=1.0e-6, 
		vinfer_iter::Integer=20,
		vinfer_tol::Float64=1.0e-8,
		fix_topics::Bool=false,
		fix_alpha::Bool=false,
		display::Symbol=:iter)

		new(int(maxiter), float64(tol), 
			int(vinfer_iter), float64(vinfer_tol), 
			fix_topics, fix_alpha,
			verbosity_level(display))
	end
end


