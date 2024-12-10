
# DeepCDR

This branch looks into the possibility of uncertainty quantification for the cancer drug response prediction models. Currently we are experimenting with the Monte-Carlo dropout method to activate dropouts during the prediction phase to get multiple predictions per given drug and omic inputs for the DeepCDR model. The experiment has been conducted for the CCLE split_0; this would be extended to other data splits, too. Later, we will use Ensemble Kalman filters to quantify uncertainty and evaluate this method's performance against the MC dropout method.
