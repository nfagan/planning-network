%%

sp = '/Users/nick/Library/CloudStorage/GoogleDrive-fagan.nicholas@gmail.com/My Drive/Documents/phd/project1/hypotheses/021725';
do_save = true;

%%  h1

%{

H1: deliberation (i.e., processing internal information) reduces 
uncertainty about how to act, at the cost of time, with this effect being 
greater for capacity-limited systems.

%}

%{

Prediction: for models trained to matching levels of average reward, 
smaller networks should see a greater reduction in policy entropy per 
additional iteration of planning.

%}

figure(1); clf;
xs = 1:10;
entropy_small = linspace( 3, 0.5, numel(xs) );
entropy_large = linspace( 1.5, 0.4, numel(xs) );

% entropy_small = exp( -linspace(0, 1, numel(xs)) * 6 ) + 0.8
% entropy_large = exp( -linspace(0, 1, numel(xs)) * 2 ) + 0.4;

plot( xs, entropy_small, 'DisplayName', 'low-capacity', 'linewidth', 2 );
hold on;
plot( xs, entropy_large, 'DisplayName', 'high-capacity', 'linewidth', 2 );
legend;
title( 'Policy entropy for increasing amounts of deliberation (rollouts)' ); 
xlabel( 'Rollout #' ); ylabel( 'entropy' );
xlim( [min(xs)-1, max(xs)+1] );
ylim( [0, 4] );

if ( do_save ), saveas(gcf, fullfile(sp, "h1_a_policy_entropy.png")); end

%{

Prediction: the slope of reduction in uncertainty per additional 
planning step should be smaller later in learning.

%}

figure(2); clf;
xs = 1:100;
slope_small = sin( (linspace(0, 0.75, numel(xs))).^(0.75)*pi ) + 0.5;
slope_large = sin( (linspace(0, 0.75, numel(xs))).^(0.50)*pi ) * 0.75 + 0.5;

plot( xs, slope_small, 'DisplayName', 'low-capacity', 'linewidth', 2 );
hold on;
plot( xs, slope_large, 'DisplayName', 'high-capacity', 'linewidth', 2 );
legend;
title( 'Reduction in policy entropy per additional step of deliberation (rollout)' ); 
xlabel( '% experience (over learning)' ); ylabel( 'Change in entropy' );
xlim( [min(xs)-1, max(xs)+1] );
ylim( [0, 2] );

if ( do_save ), saveas(gcf, fullfile(sp, "h1_b_reduction_in_policy_entropy.png")); end

%%  h2

%{

H2: deliberation enables learning systems to improve their performance by 
spending time iteratively applying limited computational resources.

%}

%{

Prediction: the smaller network will exhibit a larger change in the number 
of episodes required to reach a certain level of average reward 
(going from short → long rollouts). i.e., increased iterative processing 
should benefit the smaller network more.

%}

figure(3); clf;
xs = linspace( 0, 6, 100 );
delta_small = sin( (linspace(0, 0.75, numel(xs))).^(0.75)*pi ) * -1 * 1e3 - 300;
delta_large = sin( (linspace(0, 0.75, numel(xs))).^(0.50)*pi ) * -1 * 500 - 500;

plot( xs, delta_small, 'DisplayName', 'low-capacity', 'linewidth', 2 );
hold on;
plot( xs, delta_large, 'DisplayName', 'high-capacity', 'linewidth', 2 );
legend;
title( 'Longer (vs. shorter) rollouts benefit the low-capacity agent more' ); 
xlabel( 'Average amount of reward' ); ylabel( 'Δ episodes to reach avg. reward' );
xlim( [min(xs)-1, max(xs)+1] );

if ( do_save ), saveas(gcf, fullfile(sp, "h2_a_effect_of_rollout_length.png")); end

%%  h3

%{

H3: deliberation leads to more accurate estimates of future value.

%}

%{

Prediction: the reduction in value function error per additional rollout
should be greater for small compared to large networks, suggesting a 
stronger relationship between iterative processing and policy improvement.

%}

figure(4); clf;
xs = 1:10;
val_error_small = linspace( 3, 0.5, numel(xs) );
val_error_large = linspace( 1.5, 0.4, numel(xs) );

plot( xs, val_error_small, 'DisplayName', 'low-capacity', 'linewidth', 2 );
hold on;
plot( xs, val_error_large, 'DisplayName', 'high-capacity', 'linewidth', 2 );
legend;
title( 'Error in value function estimates for increasing amounts of deliberation (rollouts)' ); 
xlabel( 'Rollout #' ); ylabel( 'Value function error' );
xlim( [min(xs)-1, max(xs)+1] );
ylim( [0, 4] );

if ( do_save ), saveas(gcf, fullfile(sp, "h3_a_val_func_error.png")); end

%{

Prediction: this relationship should be weaker later in learning, 
reflecting that value function error decreases with experience.

%}

figure(5); clf;
xs = 1:100;
slope_small = [ -4, -3 ];
slope_large = [ -3, -1 ];
X = [slope_small; slope_large]';

bar( X );
legend( {'low-capacity', 'high-capacity'} );
set( gca, 'xticklabels', {'early in learning', 'late in learning'} );
ylabel( 'Reduction in value function error per additional rollout' );
title( sprintf(['Smaller networks continue to benefit from additional\n' ...
  , 'rollouts later in learning']) );
ylim( [min(X(:))-1, 1] );

if ( do_save ), saveas(gcf, fullfile(sp, "h3_b_val_func_error_over_learning.png")); end