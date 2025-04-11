src_p = "/Users/nick/source/mattarlab/planning-network/results";
tsrc = load_in( src_p );

%%  scalars over learning

dv = ["mean_reward", "p_plan", "vf_error", "state_pred_acc" ...
  , "reward_pred_acc", "exploit_reward_pred_acc"];
t = rowify_vars( tsrc, [dv, "epoch", "experience"] );
% t.relative_vf_error = t.vf_error ./ t.mean_reward;
% dv = ["relative_vf_error", dv];
mu = summarize_across( t, 'seed', dv, @(x) mean(x, 1) );
mu.relative_vf_error = mu.vf_error ./ mu.mean_reward;
err = summarize_across( t, 'seed', dv, @plotlabeled.sem );
l = table2str( mu(:, {'hiddens'}) );

[~, peak_plani] = max( mu.p_plan, [], 2 );

tvars = ["vf_error", "p_plan", "mean_reward" ...
  , "relative_vf_error", "state_pred_acc", "reward_pred_acc"];

figure(1); clf;
axs = plots.panels( numel(tvars) );

for idx = 1:numel(axs)
axes( axs(idx) );
tvar = tvars(idx);
colors = plot_colors( mu );
hs = plot( mu.experience', mu.(tvar)', 'linewidth', 4 );
for i = 1:rows(l), set( hs(i), 'displayname', l(i), 'color', colors(i, :) ); end
legend( hs );
ylabel( plots.strip_underscore(tvar) );
title(plots.strip_underscore(tvar));
if ( 1 )
  hold( axs(idx), 'on' );
  px = arrayfun( @(i, x) mu.experience(i, x), 1:rows(mu), peak_plani' );
  py = arrayfun( @(i, x) mu.(tvar)(i, x), 1:rows(mu), peak_plani' );
  scatter( axs(idx), px, py, 64, 'o', 'filled' );
end
if ( contains(tvar, 'acc') || contains(tvar, 'p_') ), ylim([0, 1]); end
end

plots.onelegend( gcf );

%%

function vs = vnames(t)
vs = t.Properties.VariableNames;
end

function [t, I] = rowify_vars(T, rest)
rest = string( rest );
[I, t] = rowgroups( T(:, setdiff(vnames(T), rest)) );

for i = 1:numel(rest)
  t.(rest(i)) = cate1( cellfun(@(x) reshape_var(T.(rest(i)), x), I, 'un', 0) );
end

function dv = reshape_var(v, i)
  clns = colons( ndims(v) - 1 );
  subv = v(i, clns{:});
  dv = reshape( subv, [1, size(subv)] );
end
end

function cs = plot_colors(t)
c = [0, 0, 0.8] .* [0.5; 0.75; 1];
cs = zeros( rows(t), 3 );
cs(ismember(t.hiddens, 60), :) = c(1, :);
cs(ismember(t.hiddens, 80), :) = c(2, :);
cs(ismember(t.hiddens, 100), :) = c(3, :);
end

function s = table2str(t)
s = strings( size(t, 1), 1 );
for idx = 1:size(t, 1)
  s(idx) = row2str( t(idx, :) );
end
function s = row2str(t)
  s = "";
  for i = 1:size(t, 2)
    v = compose( "%s=%s", t.Properties.VariableNames{i}, string(t{:, i}) );
    s = s + v;
    if ( i + 1 <= size(t, 2) )
      s = s + " | ";
    end
  end
end
end

function tsrc = load_in(src_p)

seeds = 61:65;
epochs = 0:100:1000;
nhid = [60, 80, 100];

tsrc = table();
for hi = 1:numel(nhid)
  for si = 1:numel(seeds)
    for ei = 1:numel(epochs)
      fname = compose( "replicate-N%d_T50_Lplan8_seed%d_%d-dst.mat" ...
        , nhid(hi), seeds(si), epochs(ei) );
      res = load( fullfile(src_p, fname) );
      va = {'p_plan', 'mean_reward', 'vf_error', 'hiddens' ...
        , 'seed', 'epoch', 'experience' ...
        , 'state_pred_acc', 'reward_pred_acc', 'exploit_reward_pred_acc'};
      t = table( ...
        res.p_plan, res.mean_total_reward, mean(res.vf_error), nhid(hi) ...
        , seeds(si), epochs(ei), epochs(ei)*200*40 ...
        , res.state_prediction_acc, res.reward_prediction_acc, res.exploit_reward_prediction_acc ...
        , 'va', va );
      tsrc(end+1, :) = t;
    end
  end
end

end