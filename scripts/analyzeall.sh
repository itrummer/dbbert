echo '' > plot.sql
for t in 2 10 100; do
	for s in pg ms; do
		echo 'drop table if exists performance; create table performance(run int, eval int, millis numeric, bestquality numeric, curquality numeric);' >> plot.sql
		inpath=ddpg/"$s"_tpch_t"$t"_performance
		echo $inpath
		command='\\'"copy performance from '$inpath' CSV header delimiter E'"\\t"';"
		echo $command >> plot.sql
		outpath=ddpg/"$s"_tpch_t"$t"_plot
		command='\\copy (With L as (with S as (select 30000*generate_series(1, 30) as millis) select S.millis as plotmillis, P.run as run, max(P.millis) as updatemillis from S, performance P where S.millis >= P.millis group by S.millis, P.run) select L.plotmillis as millis, avg(P.bestquality) as avg, 2*(case when stddev(P.bestquality) is null then 0 else stddev(P.bestquality) end) as error, min(P.bestquality) as min, max(P.bestquality) as max, percentile_cont(0.2) within group (order by P.bestquality) as p20, percentile_cont(0.8) within group (order by P.bestquality) as p80 from L, performance P where L.run = P.run and L.updatemillis = P.millis group by L.plotmillis order by L.plotmillis) to '"'$outpath' CSV header;"
		echo $command >> plot.sql
	done
done

for t in 2 10 100; do
	for s in pg ms; do
		echo 'drop table if exists performance; create table performance(run int, eval int, millis numeric, bestquality numeric, curquality numeric);' >> plot.sql
		inpath=ddpg/"$s"_tpcc_t"$t"_performance
		echo $inpath
		command='\\'"copy performance from '$inpath' CSV header delimiter E'"\\t"';"
		echo $command >> plot.sql
		outpath=ddpg/"$s"_tpcc_t"$t"_plot
		command='\\copy (With L as (with S as (select 30000*generate_series(1, 30) as millis) select S.millis as plotmillis, P.run as run, max(P.millis) as updatemillis from S, performance P where S.millis >= P.millis group by S.millis, P.run) select L.plotmillis as millis, avg(P.bestquality) as avg, 2*(case when stddev(P.bestquality) is null then 0 else stddev(P.bestquality) end) as error, min(P.bestquality) as min, max(P.bestquality) as max, percentile_cont(0.2) within group (order by P.bestquality) as p20, percentile_cont(0.8) within group (order by P.bestquality) as p80 from L, performance P where L.run = P.run and L.updatemillis = P.millis group by L.plotmillis order by L.plotmillis) to '"'$outpath' CSV header;"
		echo $command >> plot.sql
	done
done

for bench in tpcc tpch; do
	for s in pg ms; do
		echo 'drop table if exists performance; create table performance(run int, eval int, millis numeric, bestquality numeric, curquality numeric);' >> plot.sql
		inpath=dbbert/"$s"_"$bench"_base_performance
		echo $inpath
		command='\\'"copy performance from '$inpath' CSV header delimiter E'"\\t"';"
		echo $command >> plot.sql
		outpath=dbbert/"$s"_"$bench"_base_plot
		command='\\copy (With L as (with S as (select 30000*generate_series(1, 30) as millis) select S.millis as plotmillis, P.run as run, max(P.millis) as updatemillis from S, performance P where S.millis >= P.millis group by S.millis, P.run) select L.plotmillis as millis, avg(P.bestquality) as avg, 2*(case when stddev(P.bestquality) is null then 0 else stddev(P.bestquality) end) as error, min(P.bestquality) as min, max(P.bestquality) as max, percentile_cont(0.2) within group (order by P.bestquality) as p20, percentile_cont(0.8) within group (order by P.bestquality) as p80 from L, performance P where L.run = P.run and L.updatemillis = P.millis group by L.plotmillis order by L.plotmillis) to '"'$outpath' CSV header;"
		echo $command >> plot.sql
	done
done

for src in 'further_analysis/pg_tpch_by_doc_performance' 'further_analysis/pg_tpch_no_agg_performance' 'further_analysis/pg_tpch_no_implicit_performance' 'further_analysis/pg_tpch_small_performance'; do
	echo 'drop table if exists performance; create table performance(run int, eval int, millis numeric, bestquality numeric, curquality numeric);' >> plot.sql
	command='\\'"copy performance from '$src' CSV header delimiter E'"\\t"';"
	echo $command >> plot.sql
	outpath="$src"_plot
	command='\\copy (With L as (with S as (select 30000*generate_series(1, 30) as millis) select S.millis as plotmillis, P.run as run, max(P.millis) as updatemillis from S, performance P where S.millis >= P.millis group by S.millis, P.run) select L.plotmillis as millis, avg(P.bestquality) as avg, 2*(case when stddev(P.bestquality) is null then 0 else stddev(P.bestquality) end) as error, min(P.bestquality) as min, max(P.bestquality) as max, percentile_cont(0.2) within group (order by P.bestquality) as p20, percentile_cont(0.8) within group (order by P.bestquality) as p80 from L, performance P where L.run = P.run and L.updatemillis = P.millis group by L.plotmillis order by L.plotmillis) to '"'$outpath' CSV header;"
	echo $command >> plot.sql
done
