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

randdoc(model::LDAModel, len::Int) = randdoc(model, rand(model.dird), len)

# show model

function show(io::IO, model::LDAModel)
	K = ntopics(model)
	V = nterms(model)
	print(io, "LDAModel ($K topics, $V terms)")
end


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

# variational inference method

immutable LDAVarInfer
	maxiter::Int
	tol::Float64
	display::Symbol

	function LDAVarInfer(;maxiter::Integer=100, tol::Real=1.0e-4, display::Symbol=:none)
		new(int(maxiter), float64(tol), display)
	end
end

# problem & solution types

immutable LDAVarInferProblem <: IterOptimProblem
	alpha::Vector{Float64}
	tlogp::Matrix{Float64}	
	doc::SDocument

	function LDAVarInferProblem(α::Vector{Float64}, tlogp::Matrix{Float64}, doc::SDocument)
		new(α, tlogp, doc)
	end

	function LDAVarInferProblem(model::LDAModel, doc::SDocument)
		new(model.dird.alpha, model.tlogp, doc)
	end
end

ntopics(prb::LDAVarInferProblem) = length(prb.alpha)

immutable LDAVarInferSolution
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
	K = ntopics(prb)
	n = histlength(prb.doc)
	if !(length(sol.gamma) == K && size(sol.phi, 2) >= n)
		throw(ArgumentError("The LDA problem and solution are not compatible."))
	end
end

mean_theta(r::LDAVarInferSolution) = r.gamma * inv(sum(r.gamma))


# evaluation of objective

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
	K::Int = ntopics(prb)
	α::Vector{Float64} = prb.alpha
	tlogp::Matrix{Float64} = prb.tlogp
	doc::SDocument = prb.doc

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

		h_i = h[i]
		t_lpword += h_i * lpw_i
		t_φent += h_i * pent_i
	end

	# combine and return
	t_lpθ + t_lptoc + t_lpword + γent + t_φent
end


# initialize or update solution

# update solution

function update_per_gamma!(tlogp::Matrix{Float64}, doc::SDocument, r::LDAVarInferSolution)
	# update the result struct based on the gamma field

	# get model & doc fields
	K = size(tlogp, 1)
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

function initialize!(prb::LDAVarInferProblem, r::LDAVarInferSolution)
	# Inplace initialization of LDA variational inference results

	check_compatible(prb, r)

	K::Int = ntopics(prb)
	α::Vector{Float64} = prb.alpha
	tlogp::Matrix{Float64} = prb.tlogp
	doc::SDocument = prb.doc
	
	avg_tocweight::Float64 = doc.sum_counts / K
	γ = r.gamma
	for k = 1:K
		γ[k] = α[k] + avg_tocweight
	end

	update_per_gamma!(tlogp, doc, r)
end

function update!(prb::LDAVarInferProblem, sol::LDAVarInferSolution)
	# Update of one iteration

	check_compatible(prb, sol)

	K::Int = ntopics(prb)
	doc::SDocument = prb.doc
	
	# update γ
	α::Vector{Float64} = prb.alpha
	γ::Vector{Float64} = sol.gamma
	τ::Vector{Float64} = sol.tocweights

	for k = 1:K
		γ[k] = α[k] + τ[k]
	end

	# update other fields
	update_per_gamma!(prb.tlogp, doc, sol)
end

function initialize(prb::LDAVarInferProblem)
	# Create an initialized variational inference result struct

	initialize!(prb, LDAVarInferSolution(ntopics(prb), histlength(prb.doc)))
end

function infer(model::LDAModel, doc::SDocument, method::LDAVarInfer)
	solve(LDAVarInferProblem(model, doc), method.maxiter, method.tol, method.display)
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
	fix_alpha::Bool
	fix_topics::Bool
	display::Symbol

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
			fix_alpha, fix_topics, display)
	end
end

# problem and state types

immutable LDAVarLearnProblem{Corpus<:AbstractVector{SDocument}} <: IterOptimProblem
	ntopics::Int
	nterms::Int
	corpus::Corpus
end

type LDAVarLearnState
	tw::Float64                # total sample weight
	alpha::Vector{Float64}     # Dirichlet cofficients of the model
	lmnB::Float64              # log B(α)
	tlogp::Matrix{Float64}     # log of word-probablities (K x V)
	gammas::Matrix{Float64}    # a matrix of all gamma vectors

	tsol::LDAVarInferSolution    # temporary solution for current document
	vinf_objv::Float64           # accumulated objective of variational inference
	slogθ::Vector{Float64}       # accumulated E[logθ] of variational inference
	W::Matrix{Float64}           # accumulated per-topic word counts

	vinf_method::LDAVarInfer     # variational inference options
	fix_alpha::Bool
	fix_topics::Bool 
end

ntopics(st::LDAVarLearnState) = length(st.alpha)

objective(prb::LDAVarLearnProblem, st::LDAVarLearnState) = st.vinf_objv - st.tw * st.lmnB

immutable LDAVarLearnResults
	model::LDAModel
	gammas::Matrix{Float64}
end

function get_results(st::LDAVarLearnState)
	topics = normalize!(exp!(st.tlogp'), 1, 1)
	model = LDAModel(st.alpha, topics)
	LDAVarLearnResults(model, st.gammas)	
end


# initialize

function initialize(prb::LDAVarLearnProblem, method::LDAVarLearn, model::LDAModel, gammas::Matrix{Float64})

	corpus = prb.corpus
	ndocs = length(corpus)
	K = ntopics(model)
	V = nterms(model)
	size(gammas) == (K, ndocs) || throw(ArgumentError("The size of gammas is incorrect."))

	# prepare the temporary solution structure
	max_hlen = max_histlength(corpus)
	tsol = LDAVarInferSolution(K, max_hlen)

	# prepare other data structures
	slogθ = Array(Float64, K)
	W = Array(Float64, K, V)

	# method
	vinf_method = LDAVarInfer(maxiter=method.vinfer_iter, tol=method.vinfer_tol, display=:none)

	# initial model
	alpha = model.dird.alpha
	lmnB = model.dird.lmnB
	tlogp = model.tlogp

	LDAVarLearnState(float64(ndocs), alpha, lmnB, tlogp, gammas, tsol, NaN, slogθ, W, 
		vinf_method, method.fix_alpha, method.fix_topics)
end

function initialize(prb::LDAVarLearnProblem, method::LDAVarLearn, model::LDAModel)
	# initialize gamma
	corpus = prb.corpus
	K::Int = prb.ntopics
	if ntopics(model) != K
		throw(ArgumentError("The numbers of topics do not match."))
	end

	ndocs::Int = length(corpus)
	α::Vector{Float64} = model.dird.alpha

	gammas = Array(Float64, K, ndocs)
	for i = 1:ndocs
		doc::SDocument = corpus[i]
		avg_tocweight::Float64 = doc.sum_counts / K
		for k = 1:K
			gammas[k,i] = α[k] + avg_tocweight
		end
	end

	# construct the entire state
	initialize(prb, method, model, gammas)
end


# Update

function update!{Corpus}(prb::LDAVarLearnProblem{Corpus}, st::LDAVarLearnState)
	perform_varinfer!(st, prb.corpus)

	if !st.fix_alpha
		update_alpha!(st)
	end

	if !st.fix_topics
		update_topics!(st)
	end
end

function perform_varinfer!(st::LDAVarLearnState, corpus::AbstractVector{SDocument})
	K::Int = ntopics(st)

	# reset accumulators
	objv = 0.
	slogθ = st.slogθ
	fill!(slogθ, 0.)
	W = st.W
	fill!(W, 0.)

	# prepare temporary solution
	alpha::Vector{Float64} = st.alpha
	tlogp::Matrix{Float64} = st.tlogp
	gammas::Matrix{Float64} = st.gammas
	tsol::LDAVarInferSolution = st.tsol
	γ::Vector{Float64} = tsol.gamma
	vinf_method = st.vinf_method

	for i = 1:length(corpus)
		doc::SDocument = corpus[i]

		# resume the temporary state
		for k = 1:K
			@inbounds γ[k] = gammas[k,i]
		end
		update_per_gamma!(tlogp, doc, tsol)

		# perform inference
		info = iter_optim!(LDAVarInferProblem(alpha, tlogp, doc), tsol, vinf_method.maxiter, vinf_method.tol)

		# collect relevant statistics
		add!(slogθ, tsol.elogtheta)
		add_word_counts!(W, doc.terms, doc.counts, tsol.phi)
		objv += info.objective
	end
	st.vinf_objv = objv
end

function update_alpha!(st::LDAVarLearnState)
	slogθ = multiply!(st.slogθ, inv(st.tw))
	Distributions.fit_mle!(Dirichlet, slogθ, st.alpha)

	α = st.alpha
	K = length(α)
	lmnB = 0.
	α0 = 0.
	for k = 1:K
		αk = α[k]
		α0 += αk
		lmnB += lgamma(αk)
	end
	lmnB -= lgamma(α0)
	st.lmnB = lmnB
end

function update_topics!(st::LDAVarLearnState)
	tlogp = st.tlogp
	log!(normalize!(tlogp, st.W, 1, 2))
end

function learn(corpus::AbstractVector{SDocument}, init_model::LDAModel, method::LDAVarLearn)
	K = ntopics(init_model)
	V = nterms(init_model)
	prb = LDAVarLearnProblem(K, V, corpus)

	st = initialize(prb, method, init_model)
	info = iter_optim!(prb, st, method.maxiter, method.tol, method.display)
	return (get_results(st), info)
end

