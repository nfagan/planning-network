root_p = "/Users/nick/source/mattarlab/planning_code/computational_model/results";
rew_plan_res = load( fullfile(root_p, "rew_and_plan_by_n.mat") );

policies = table();
for epoch = 0:100:1000
  for unit = [60, 100]
    fname = compose( "perf_by_n_epoch_%d_N%d_Lplan8.mat", epoch, unit );
    rollout_res = load( fullfile(root_p, fname) );
    policy_t = rollout_res_policies_to_table( rollout_res );
    policy_t.units(:) = unit;
    policy_t.epoch(:) = epoch;
    policies = [ policies; policy_t ];
  end
end

%%

%{
  res.planfracs (ihid, iseed, iepoch)
%}

episodes_per_epoch = 40 * 200; %  @TODO: This was fixed during training.

p_plan = squeeze( mean(rew_plan_res.planfracs, 2) ); %  mean across seeds
mu_entropy = squeeze( mean(rew_plan_res.mu_entropies, 2) ); %  mean across seeds
tot_episodes = rew_plan_res.epochs * episodes_per_epoch;

figure(1); clf;
subplot( 1, 2, 1 );
plot_by_model_capacity( tot_episodes, p_plan, rew_plan_res.Nhiddens );
title( 'p(plan)' ); ylim( [0, 1] );
subplot( 1, 2, 2 );
plot_by_model_capacity( tot_episodes, mu_entropy, rew_plan_res.Nhiddens );
title( 'policy entropy' ); ylim( [0, 200] );

%%

% Ignore planning action (5)
ps = softmax( policies.policy(:, 1:4), 2 );
entropies = -sum( ps .* log(ps), 2 );

[I, C] = findeach( policies, setdiff(vnames(policies) ...
  , {'policy', 'replication', 'seed'}) );
mu_entropies = rowifun( @nanmean, I, entropies );

% organize as time course over planning steps
[I, entropy_t] = findeach( C, setdiff(vnames(C), {'ith_plan_step'}) );
entropy_t.entropy = cate1( cellfun(@(x) mu_entropies(x)', I, 'un', 0) );
entropy_t.ith_plan_step = cate1( cellfun(@(x) C.ith_plan_step(x)', I, 'un', 0) );

%%

sliced_epoch = 1e3;
slice_mask = ...
    entropy_t.epoch == sliced_epoch ...
  & entropy_t.ith_action_to_goal == 1 ...
  & ~entropy_t.planning_disabled;

slice_subset = entropy_t(slice_mask, :);

figure(1); clf;
h = plot( slice_subset.ith_plan_step', slice_subset.entropy', 'linewidth', 2 );
h_lab = table2str( slice_subset(:, {'units', 'epoch'}) );
arrayfun( @(h, s) set(h, 'displayname', s), h, h_lab );
legend( h );
xlabel( 'Rollout #' );
ylabel( 'Entropy' );

for i = 1:size(slice_subset, 1)
  % mdl = @(b,x)b(1).*exp(b(2).*x)+b(3);
  % x0 = [20,.001,0];

  % mdl = @(b,x)exp(b(1).*x)+b(2);
  % x0 = [-10,0];

  mdl = @(b,x)b(1).*exp(-2e-1*x)+b(2);
  x0 = [1,0];

  x = slice_subset.ith_plan_step(i, :);
  f = fitnlm(x, slice_subset.entropy(i, :), mdl, x0);
  b1 = f.Coefficients.Estimate(1);

  hold on;
  hf = plot( x, predict(f, x(:)), 'LineWidth', 4, 'LineStyle', '--' );
  set( hf, 'color', get(h(i), 'color'), 'displayname', compose("beta = %0.3f", b1) );
end

title( compose("Epoch = %d", sliced_epoch) );

%%

epochs = 0:100:1000;
slope_ts = table();

for ep = epochs

slice_mask = ...
    entropy_t.epoch == ep ...
  & entropy_t.ith_action_to_goal == 1 ...
  & ~entropy_t.planning_disabled;

slice_subset = entropy_t(slice_mask, :);
slice_subset.slopes(:) = nan;

for i = 1:size(slice_subset, 1)
  mdl = @(b,x)b(1).*exp(-2e-1*x)+b(2);
  x0 = [1,0];

  x = slice_subset.ith_plan_step(i, :);
  f = fitnlm( x, slice_subset.entropy(i, :), mdl, x0 );
  slice_subset.slopes(i) = f.Coefficients.Estimate(1);
end

slope_ts = [ slope_ts; slice_subset ];

end

figure(1); clf;
g1 = slope_ts.units == 60;
g2 = slope_ts.units == 100;
x = [ slope_ts.epoch(g1)'; slope_ts.epoch(g2)' ];
x = x ./ max(x(:)) * 100;

y = [ slope_ts.slopes(g1)'; slope_ts.slopes(g2)' ];
h = plot( x', y', 'linewidth', 2 );
s = table2str( slope_ts([find(g1, 1), find(g2, 1)], {'units'}) );
arrayfun( @(h, s) set(h, 'displayname', s), h, s );
legend( h );
xlabel( '% Experience' );
ylabel( 'Slope of exp. decay in entropy per additional rollout' );

%%

figure(1); clf;
num_rollouts = 0:15;
rollout_subsets = 1:5:numel(num_rollouts);

[I, C] = findeach( entropy_t, {'units', 'planning_disabled', 'ith_action_to_goal'} ...
  , ismember(entropy_t.ith_action_to_goal, [1]) );
axs = plots.panels( numel(I) );

for axi = 1:numel(I)

ax = axs(axi);
subset = entropy_t(I{axi}, :);
h = plot( ax, subset.epoch, subset.entropy(:, rollout_subsets), 'LineWidth', 2 );
colors = spring( numel(h) );
names = compose( "Rollout = %d", num_rollouts(rollout_subsets) );
for i = 1:numel(h), set(h(i), 'color', colors(i, :), 'displayname', names(i)); end
legend( h );
xlabel( ax, 'Training epoch' );
ylabel( ax, 'entropy (nats)' );

s = plots.strip_underscore( table2str(C(axi, :)) );
title( ax, s );
ylim( ax, [0, 2] );

end

%%

function plot_by_model_capacity(x, y, nhid)
h = plot( x, y, 'linewidth', 2 );
arrayfun( @(h, x) set(h, 'DisplayName', x), h, compose("Hidden = %d", nhid) );
legend( h );
xlabel( 'Total # training episodes' );
end

function ts = rollout_res_policies_to_table(res)

% policies = zeros(2, nreps, length(nplans), 10, 5) .+ NaN;

fs = fieldnames( res );
policy_fs = fs(contains(fs, 'policies'));
second = @(x) string(x{2});
seeds = cellfun( ...
  @(x) extract(second(strsplit(x, 'seed_')), digitsPattern), policy_fs );
seeds = double( seeds );

pds = logical.empty;
nrs = [];
nps = [];
t_seeds = [];
slices = [];
nas = [];

for i = 1:numel(policy_fs)
  fprintf( '\n\t %d of %d', i, numel(policy_fs) );

  v = res.(policy_fs{i});
  nreps = size( v, 2 );
  nplan = 0:15; % @TODO: This was fixed during evaluation
  nact_to_goal = size( v, 4 ); % action number to goal
  num_actions = size( v, 5 );
  seed = seeds(i);

  for ic = 1:size(v, 1)
    planning_disabled = ic == 2;
    for nr = 1:nreps
      for np = 1:numel(nplan)
        for na = 1:nact_to_goal
          slice = squeeze( v(ic, nr, np, na, :) )';
          slices(end+1, :) = slice;
          pds(end+1, :) = planning_disabled;
          nrs(end+1, :) = nr;
          nps(end+1, :) = nplan(np);
          nas(end+1, :) = na;
          t_seeds(end+1, :) = seed;
        end
      end
    end
  end
end

ts = table( slices, pds, nrs, nps, nas, t_seeds ...
            , 'va', {'policy', 'planning_disabled' ...
            , 'replication', 'ith_plan_step', 'ith_action_to_goal', 'seed'} );

end

function v = vnames(t)
v = t.Properties.VariableNames;
end

function y = softmax(x, varargin)
y = exp( x ) ./ sum( exp(x), varargin{:} );
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