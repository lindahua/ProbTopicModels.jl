# Test common types

using ProbTopicModels
using Base.Test

# vocabulary 

vlist = ["math", "chem", "phy", "sci", "eng"]
voc = vocabulary(vlist)

@test length(voc) == length(vlist)
@test voc.terms === vlist

for i in 1 : length(vlist)
	@test encode(voc, vlist[i]) == i
end

@test encode(voc, vlist) == [1:length(vlist)]

# simple documents

sdoc = SDocument([1, 2, 3, 4], [3, 2, 1, 5])
@test sdoc.terms == [1, 2, 3, 4]
@test sdoc.counts == [3., 2., 1., 5.]
@test sdoc.sum_counts == 11.

ts = [1, 1, 1, 3, 2, 2, 3, 3, 3, 2, 3, 4, 3]
sdoc2 = SDocument(ts)
@test sdoc2.terms == [1, 2, 3, 4]
@test sdoc2.counts == [3., 3., 6., 1.]
@test sdoc2.sum_counts == 13.

@test count_words(4, [sdoc, sdoc2]) == [6., 5., 7., 6.]
