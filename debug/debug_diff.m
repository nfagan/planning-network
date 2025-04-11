% entr = @(p) -sum(p .* log(p));
nrm = @(x) x ./ sum(x);
entr(nrm([1, 1]))
entr(nrm([1, 1, 1]))
entr(nrm([1, 1, 1, 1, 1]))

m = fminsearch( @(x) -entr(x, true), [1, 1, 0, 1, 1] );
entr( m, true )

%%

src_p = '/Users/nick/source/mattarlab/planning_code/computational_model/dump_models';
mats = shared_utils.io.find( src_p, '.mat' );
mats(contains(mats, '-dst')) = [];

for i = 1:numel(mats)
  fprintf( '\n %d of %d', i, numel(mats) );
  src = load( mats{i} );
  [~, fname, ext] = fileparts( mats{i} );
  dst_p = fullfile( src_p, compose("%s-dst.mat", fname) );
  save( dst_p, '-struct', 'src' );
end

%%

x = load('/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/walls.mat');
w = [x.walls(:, 1, :); x.walls(:, 3, :)];
w = squeeze(w);
w = w';

%%

x = load('/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/walls.mat');
save('/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/walls.mat-dst.mat','-struct','x')

x = load('/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/output_N100_T50_Lplan8_seed62_1000.mat');
save('/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/output_N100_T50_Lplan8_seed62_1000-dst.mat','-struct','x')

%%

y = load( '/Users/nick/source/mattarlab/planning_code/computational_model/dump_models/my_output-dst.mat' );

bi = 1;
ti = 6;
kx = x.agent_inputs(:, bi, ti);
mx = y.xs(bi, :, ti);
comp = [kx(:), mx(:)];

nt = min( size(x.as, 2), size(y.actions, 2) );

xt = x.agent_inputs(7, bi, 1:nt);
yt = y.xs(bi, 7, 1:nt);

xh = squeeze( x.hs(:, bi, 1:nt) );
yh = squeeze( y.hs(bi, :, 1:nt) );
dh = max( abs(xh - yh), [], 1 );

ahot = comp(1:5, :);
prev_rewards = comp(6, :);
time = comp(7, :);
shot = comp(8:8+15, :);
walls = comp(8+16:8+16*3-1, :);
plan_input = comp(8+16*3:end, :);

hs = [ x.hs(:, bi, ti), y.hs(bi, :, ti)' ];
as = [ x.as(bi, ti), y.actions(bi, ti)+1 ];

ts = [ columnize(xt), columnize(yt) ];
tot = [ double(x.as(bi, 1:nt)'), double(y.actions(bi, 1:nt)'+1), ts ];

if ( 1 ), hs = abs(diff(hs, 1, 2)); end;
if ( 0 ), time = abs(diff(time, 1, 2)); end;

yplan = sum(y.actions(:) == 4 & y.actives(:)) / sum(y.actives(:));

% x[:, :meta.num_actions] = prev_ahot
%   x[:, meta.num_actions:meta.num_actions+1] = prev_rewards
%   # x[:, meta.num_actions+1:meta.num_actions+2] = time / meta.T
%   x[:, meta.num_actions+1:meta.num_actions+2] = time / 50.0; assert meta.T < 50.0 # walls.jl/gen_input
%   x[:, meta.num_actions+2:meta.num_actions+2+meta.num_states] = shot
%   x[:, meta.num_actions+2+meta.num_states:meta.num_actions+2+meta.num_states+meta.num_states*2] = walls
%   if plan_input is not None:
%     x[:, meta.num_actions+2+meta.num_states+meta.num_states*2:] = plan_input

%%

function h = entr(p, do_norm)
if ( nargin < 2 ), do_norm = false; end
if ( do_norm ), p = p ./ sum( p ); end
ht = p .* log(p);
ht(p == 0) = 0;
h = -sum( ht );
end