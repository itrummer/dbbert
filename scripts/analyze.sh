echo '' > plot.sql
for t in 2 10 100; do
	for s in pg ms; do
		echo 'drop table if exists performance; create table performance(run int, eval int, millis numeric, bestquality numeric, curquality numeric);' >> plot.sql
		inpath=ddpg/"$s"_tpch_t"$t"_performance
		echo $inpath
		command='\\'"copy performance from '$inpath' CSV header delimiter E'"\\t"';"
		echo $command >> plot.sql
		outpath=ddpg/"$s"_tpch_t"$t"_plot
		command='\\copy (with S as (select 30000*generate_series(1, 30) as millis) select S.millis, avg(P.bestquality) as avg, 2*stddev(P.bestquality) as error, avg(P.bestquality) - 2*stddev(P.bestquality) as lb, avg(P.bestquality) + 2*stddev(P.bestquality) as ub from S, performance P where P.millis <= S.millis group by S.millis order by S.millis) to '"'$outpath' CSV header;"
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
		command='\\copy (with S as (select 30000*generate_series(1, 30) as millis) select S.millis, avg(P.bestquality) as avg, 2*stddev(P.bestquality) as error, avg(P.bestquality) - 2*stddev(P.bestquality) as lb, avg(P.bestquality) + 2*stddev(P.bestquality) as ub from S, performance P where P.millis <= S.millis group by S.millis order by S.millis) to '"'$outpath' CSV header;"
		echo $command >> plot.sql
	done
done