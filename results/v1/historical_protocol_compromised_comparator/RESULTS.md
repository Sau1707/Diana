# Historical protocol-compromised custom comparator

This directory preserves aggregate evidence for `joint_bayes_personalizer`. The implementation selected a global diagonal/full covariance mode using fold-0 validation participants; that participant group later served as outer test in another fold. These results are therefore legacy descriptive evidence only and do not appear in the active Diana-H3P leaderboard.

Historical aggregate scores were 0.642639 in cold start and 0.640382, 0.611701, and 0.605710 on the common K=0/3/7 suffix. The model did not pass its prespecified validation superiority gate. No statistical, clinical, or confirmatory superiority claim is supported.

The source remains under `model/joint_bayes_personalizer/`. Active results are generated under `results/v1/diana_h3p/`.
