src="/tmp/performance"
echo 'drop table if exists performance; create table performance(run int, eval int, millis numeric, bestquality numeric, curquality numeric);' >> plot.sql
command='\\'"copy performance from '$src' CSV header delimiter E'"\\t"';"
echo $command >> plot.sql
outpath="$src"_plot
command='\\copy (With L as (with S as (select 30000*generate_series(1, 50) as millis) select S.millis as plotmillis, P.run as run, max(P.millis) as updatemillis from S, performance P where S.millis >= P.millis group by S.millis, P.run) select L.plotmillis as millis, avg(P.bestquality) as avg, 2*(case when stddev(P.bestquality) is null then 0 else stddev(P.bestquality) end) as error, min(P.bestquality) as min, max(P.bestquality) as max, percentile_cont(0.2) within group (order by P.bestquality) as p20, percentile_cont(0.8) within group (order by P.bestquality) as p80 from L, performance P where L.run = P.run and L.updatemillis = P.millis group by L.plotmillis order by L.plotmillis) to '"'$outpath' CSV header;"
echo $command >> plot.sql
