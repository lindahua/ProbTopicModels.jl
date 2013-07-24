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
	display::Symbol

	function LDAVarInfer(;maxiter::Integer=100, tol::Real=1.0e-4, display::Symbol=:none)
		new(int(maxiter), float64(tol), display)
	end
end

iter_options(m::LDAVarInfer) = IterOptimOptions(maxiter=m.maxiter, tol=m.tol, display=m.display)

immutable LDAVarInferProblem <: IterOptimProblem
	model::LDAModel
	doc::SDocument
end

immutable LDAVarInferSolution <: IterOptimSolution
	gamma::Vector{Float64}
	elogtheta::Vector{Float64}
	phi::Matrix{Float64}
	tocweights::Vector{Float64}

	function LDAVarInferSolution(K::Int, nmax::Int)
		new(Array(Float64, K), 
			Array(Float64, K), 
			Array(Float64, K, nmax), 
			Array(Float64, K))
	end
end

function check_compatible(prb::LDAVarInferProblem, sol::LDAVarInferSolution)
	K = ntopics(prb.model)
	n = histlength(prb.doc)
	if !(length(sol.gamma) == K && size(sol.phi, 2) >= n)
		throw(ArgumentError("The LDA problem and solution are not compatible."))
	end
end

mean_theta(r::LDAVarInferSolution) = r.gamma * inv(sum(r.gamma))

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


function objective(prb::LDAVarInferProblem, sol::LDAVarInferSolution)
	# compute the objective of LDA variational inference

	# problem fields
	model::LDAModel = prb.model
	doc::SDocument = prb.doc

	K::Int = ntopics(model)
	α::Vector{Float64} = model.dird.alpha
	tlogp::Matrix{Float64} = model.tlogp

	n::Int = histlength(doc)
	terms::Vector{Int} = doc.terms
	h::Vector{Float64} = doc.counts

	# solution fields

	γ = sol.gamma
	elogθ = sol.elogtheta
	φ = sol.phi
	τ = sol.tocweights

	# evaluation of individual terms

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

	# combine and return
	t_lpθ + t_lptoc + t_lpword + γent + t_φent
end


function update_per_gamma!(model::LDAModel, doc::SDocument, r::LDAVarInferSolution)
	# update the result struct based on the gamma field

	K::Int = ntopics(model)

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

function update!(prb::LDAVarInferProblem, sol::LDAVarInferSolution)
	# Update of one iteration

	check_compatible(prb, sol)

	model::LDAModel = prb.model
	doc::SDocument = prb.doc
	K::Int = ntopics(model)

	# update γ
	α::Vector{Float64} = model.dird.alpha
	γ::Vector{Float64} = sol.gamma
	τ::Vector{Float64} = sol.tocweights

	for k = 1:K
		γ[k] = α[k] + τ[k]
	end

	# update other fields
	update_per_gamma!(model, doc, sol)
end

function initialize!(prb::LDAVarInferProblem, r::LDAVarInferSolution)
	# Inplace initialization of LDA variational inference results

	check_compatible(prb, r)

	model::LDAModel = prb.model
	doc::SDocument = prb.doc
	K::Int = ntopics(model)
	α::Vector{Float64} = model.dird.alpha

	avg_tocweight::Float64 = doc.sum_counts / K
	γ = r.gamma
	for k = 1:K
		γ[k] = α[k] + avg_tocweight
	end

	update_per_gamma!(model, doc, r)
end


function initialize(prb::LDAVarInferProblem)
	# Create an initialized variational inference result struct

	initialize!(prb, LDAVarInferSolution(ntopics(prb.model), histlength(prb.doc)))
end

infer(model::LDAModel, doc::SDocument, method::LDAVarInfer) = solve(LDAVarInferProblem(model, doc), iter_options(method))


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

immutable LDAVarLearnProblem
	ntopics::Int
	nterms::Int
	corpus::Vector{SDocument}
end

type LDAVarLearnSolution
	model::LDAModel
	gammas::Matrix{Float64}  # each column for a document

	# Note: fields below are only for internal use

	_tw::Float64             		# total sample weights
	_vinfer_objv::Float64    		# sum of variational inference objective
	_sum_elogθ::Vector{Float64}     # sum of E[log(θ)] from all documents
	_wcounts::Matrix{Float64}       # total per-topic word counts

	_temp_vinfer_sol::LDAVarInferSolution  # temporary inference solution 

	function LDAVarLearnSolution(model::LDAModel, gammas::Matrix{Float64}, maxhistlen::Int)
		K = ntopics(model)
		V = nterms(model)
		if size(gammas, 1) != K
			throw(ArgumentError("The size of gammas does not match the model."))
		end

		K = ntopics(model)
		tw = 0.
		vinf_objv = NaN
		sum_elogθ = Array(Float64, K)
		wcounts = Array(Float64, K, V)
		vinf_sol = LDAVarInferSolution(K, maxhistlen)

		new(model, gammas, tw, vinf_objv, sum_elogθ, wcounts, vinf_sol)
	end
end

objective(prb::LDAVarLearnProblem, sol::LDAVarLearnSolution) = sol._vinfer_objv

function initialize(prb::LDAVarLearnProblem, model::LDAModel, gammas::Matrix{Float64})
	K = prb.ntopics
	V = prb.nterms
	corpus = prb.corpus
	ndocs = length(corpus)

	if ntopics(model) != K || nterms(model) != V
		throw(ArgumentError("The size of model does not match the problem."))
	end

	if !(size(gammas) != (K, ndocs))
		throw(ArgumentError("The size of gammas does not match the problem."))
	end

	LDAVarLearnSolution(model, gammas, max_histlength(corpus))
end

function initialize(prb::LDAVarLearnProblem, model::LDAModel)
	K = size(model, 1)
	corpus = prb.corpus
	ndocs = length(corpus)

	# initialize gammas
	α::Vector{Float64} = prb.model.dird.alpha
	gammas = Array(Float64, K, ndocs)
	for i = 1:ndocs
		doc::SDocument = corpus[i]
		avg_tocweight::Float64 = doc.sum_counts / K
		for k = 1:K
			@inbounds gammas[k,i] = α[k] + avg_tocweight
		end
	end

	initialize(prb, model, gammas)
end

function perform_inference!(sol::LDAVarLearnSolution, corpus::AbstractVector{SDocument}, opts::IterOptimOptions)
	model::LDAModel = sol.model
	K::Int = ntopics(model)
	n::Int = length(corpus)
	gammas = sol.gammas
	@assert size(gammas) == (K, n)
 
	# reset 
	tw = 0.
	objv = 0.
	slogθ = sol._sum_elogθ
	fill!(slogθ, 0.)
	W = sol._wcounts
	fill!(W, 0.)

	inf_sol = sol._temp_vinfer_sol
	γ::Vector{Float64} = inf_sol.gamma

	# perform per-document inference
	for i = 1:n
		doc::SDocument = corpus[i]

		# copy the temporary gamma
		for k = 1:K
			@inbounds γ[k] = gammas[k,i]
		end

		# initialize inference
		update_per_gamma!(model, doc, inf_sol)

		# inference updating loops
		r = iter_optim!(LDAVarInferProblem(model, doc), inf_sol, opts)

		# collect relevant statistics
		tw += 1.0
		objv += r.objective
		add!(slogθ, inf_sol.elogtheta)
		add_word_counts!(W, doc.terms, doc.counts, inf_sol.phi)
	end

	sol._tw = tw
	sol._vinfer_objv = objv
end

function update_model!(sol::LDAVarLearnSolution, fix_dird::Bool, fix_topics::Bool)
	model::LDAModel = sol.model
	dird::Dirichlet = model.dird
	topics::Matrix{Float64} = model.topics

	# Update Dirichlet distribution
	if !fix_dird
		elogθ = multiply!(sol._sum_elogθ, inv(sol._tw))
		dird = Distributions.fit_mle!(Dirichlet, elogθ, dird.alpha)
	end

	# Update topics
	if !fix_topics
		topics = transpose(sol._wcounts)
		V = size(topics, 1)
		K = size(topics, 2)

		for k = 1:K
			s = 0.
			for i=1:V
				s += topics[i,k]
			end
			inv_s = inv(s)
			for i=1:V
				topics[i,k] *= inv_s
			end
		end
	end

	sol.model = LDAModel(dird, topics)
end


