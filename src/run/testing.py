'''
Created on Apr 14, 2021

@author: immanueltrummer
'''
from doc.collection import DocCollection
from dbms.postgres import PgConfig
from dbms.mysql import MySQLconfig
from all.agents import VQN
from all.approximation import QNetwork
from all.environments import GymEnvironment
from all.experiments import run_experiment
from all.logging import DummyWriter
from all.policies.greedy import GreedyPolicy
from torch.optim import Adam
from torch import nn
from all.presets.classic_control import dqn
from environment.multi_doc import MultiDocTuning
from environment.supervised import LabeledDocTuning
from environment.hybrid import HybridDocTuning
from benchmark.evaluate import OLAP, TpcC
from search.objectives import Objective
from search.search_with_hints import ParameterExplorer

# dbms = MySQLconfig('tpch', 'root', 'mysql1234-', '/usr/local/mysql/bin')
# dbms.reset_config()
# # good_conf = {'admin_port': '65701', 'auto_increment_increment': '4', 'auto_increment_offset': '2', 'back_log': '1005', 'binlog_cache_size': '161963', 'binlog_expire_logs_seconds': '11949727', 'binlog_group_commit_sync_delay': '0', 'binlog_group_commit_sync_no_delay_count': '0', 'binlog_max_flush_queue_time': '0', 'binlog_row_event_max_size': '49331', 'binlog_stmt_cache_size': '34974', 'binlog_transaction_compression_level_zstd': '22', 'binlog_transaction_dependency_history_size': '157669', 'bulk_insert_buffer_size': '35390413', 'connect_timeout': '80', 'cte_max_recursion_depth': '7658', 'default_password_lifetime': '0', 'default_week_format': '0', 'delayed_insert_limit': '81', 'delayed_insert_timeout': '2073', 'delayed_queue_size': '288', 'div_precision_increment': '14', 'eq_range_index_dive_limit': '903', 'expire_logs_days': '0', 'flush_time': '0', 'ft_max_word_len': '263', 'ft_min_word_len': '16', 'ft_query_expansion_limit': '105', 'generated_random_password_length': '194', 'group_concat_max_len': '4105', 'gtid_executed_compression_period': '0', 'histogram_generation_max_mem_size': '45007024', 'host_cache_size': '1144', 'information_schema_stats_expiry': '677449', 'innodb_adaptive_flushing_lwm': '43', 'innodb_adaptive_hash_index_parts': '56', 'innodb_adaptive_max_sleep_delay': '219307', 'innodb_api_bk_commit_interval': '1', 'innodb_api_trx_level': '0', 'innodb_autoextend_increment': '206', 'innodb_autoinc_lock_mode': '4', 'innodb_buffer_pool_chunk_size': '429939869', 'innodb_buffer_pool_dump_pct': '54', 'innodb_buffer_pool_instances': '0', 'innodb_buffer_pool_size': '1267902678', 'innodb_change_buffer_max_size': '210', 'innodb_commit_concurrency': '0', 'innodb_compression_failure_threshold_pct': '26', 'innodb_compression_level': '51', 'innodb_compression_pad_pct_max': '16', 'innodb_concurrency_tickets': '47059', 'innodb_doublewrite_batch_size': '0', 'innodb_doublewrite_files': '17', 'innodb_doublewrite_pages': '31', 'innodb_fast_shutdown': '8', 'innodb_fill_factor': '714', 'innodb_flush_log_at_timeout': '2', 'innodb_flush_log_at_trx_commit': '5', 'innodb_flush_neighbors': '0', 'innodb_flushing_avg_loops': '183', 'innodb_force_recovery': '0', 'innodb_fsync_threshold': '0', 'innodb_ft_cache_size': '25499390', 'innodb_ft_max_token_size': '26', 'innodb_ft_min_token_size': '5', 'innodb_ft_num_word_optimize': '8620', 'innodb_ft_result_cache_limit': '6737738890', 'innodb_ft_sort_pll_degree': '12', 'innodb_ft_total_cache_size': '1724293620', 'innodb_idle_flush_pct': '874', 'innodb_io_capacity': '1891', 'innodb_io_capacity_max': '53688856389476859904', 'innodb_lock_wait_timeout': '52', 'innodb_log_buffer_size': '6293619', 'innodb_log_file_size': '173554868', 'innodb_log_files_in_group': '10', 'innodb_log_spin_cpu_abs_lwm': '495', 'innodb_log_spin_cpu_pct_hwm': '197', 'innodb_log_wait_for_flush_spin_hwm': '3650', 'innodb_log_write_ahead_size': '48894', 'innodb_lru_scan_depth': '10006', 'innodb_max_dirty_pages_pct': '210', 'innodb_max_dirty_pages_pct_lwm': '53', 'innodb_max_purge_lag': '0', 'innodb_max_purge_lag_delay': '0', 'innodb_max_undo_log_size': '1283551792', 'innodb_old_blocks_pct': '6', 'innodb_old_blocks_time': '2328', 'innodb_online_alter_log_max_size': '1224680031', 'innodb_open_files': '11534', 'innodb_page_cleaners': '4', 'innodb_page_size': '7556', 'innodb_parallel_read_threads': '11', 'innodb_purge_batch_size': '2620', 'innodb_purge_rseg_truncate_frequency': '189', 'innodb_purge_threads': '1', 'innodb_read_ahead_threshold': '550', 'innodb_read_io_threads': '30', 'innodb_replication_delay': '0', 'innodb_rollback_segments': '1034', 'innodb_sort_buffer_size': '9424174', 'innodb_spin_wait_delay': '14', 'innodb_spin_wait_pause_multiplier': '293', 'innodb_stats_persistent_sample_pages': '4', 'innodb_stats_transient_sample_pages': '65', 'innodb_sync_array_size': '1', 'innodb_sync_spin_loops': '186', 'innodb_thread_concurrency': '0', 'innodb_thread_sleep_delay': '46052', 'innodb_undo_tablespaces': '1', 'innodb_write_io_threads': '2', 'interactive_timeout': '274980', 'join_buffer_size': '2009927', 'key_buffer_size': '72496179', 'key_cache_age_threshold': '264', 'key_cache_block_size': '9890', 'key_cache_division_limit': '387', 'large_page_size': '0', 'lock_wait_timeout': '311508000', 'log_error_verbosity': '5', 'log_throttle_queries_not_using_indexes': '0', 'long_query_time': '34', 'lower_case_table_names': '8', 'max_allowed_packet': '280013135', 'max_binlog_cache_size': '5940758920576284672', 'max_binlog_size': '1886907796', 'max_binlog_stmt_cache_size': '156126599719175421952', 'max_connect_errors': '776', 'max_connections': '1457', 'max_delayed_threads': '161', 'max_digest_length': '7517', 'max_error_count': '669', 'max_execution_time': '0', 'max_heap_table_size': '28199161', 'max_insert_delayed_threads': '136', 'max_join_size': '153572582857097543680', 'max_length_for_sort_data': '33326', 'max_points_in_geometry': '550906', 'max_prepared_stmt_count': '86276', 'max_relay_log_size': '0', 'max_seeks_for_key': '130167782620219785216', 'max_sort_length': '4225', 'max_sp_recursion_depth': '0', 'max_user_connections': '0', 'max_write_lock_count': '175280457596545269760', 'min_examined_row_limit': '0', 'myisam_data_pointer_size': '10', 'myisam_max_sort_file_size': '16336711651491684352', 'myisam_mmap_size': '79424882532278288384', 'myisam_repair_threads': '3', 'myisam_sort_buffer_size': '28680830', 'mysqlx_connect_timeout': '124', 'mysqlx_deflate_default_compression_level': '28', 'mysqlx_deflate_max_client_compression_level': '5', 'mysqlx_document_id_unique_prefix': '0', 'mysqlx_idle_worker_thread_timeout': '463', 'mysqlx_interactive_timeout': '245524', 'mysqlx_lz4_default_compression_level': '7', 'mysqlx_lz4_max_client_compression_level': '29', 'mysqlx_max_allowed_packet': '367525230', 'mysqlx_max_connections': '897', 'mysqlx_min_worker_threads': '13', 'mysqlx_port': '129322', 'mysqlx_port_open_timeout': '0', 'mysqlx_read_timeout': '278', 'mysqlx_wait_timeout': '181381', 'mysqlx_write_timeout': '304', 'mysqlx_zstd_default_compression_level': '28', 'mysqlx_zstd_max_client_compression_level': '77', 'net_buffer_length': '117823', 'net_read_timeout': '41', 'net_retry_count': '76', 'net_write_timeout': '366', 'ngram_token_size': '3', 'open_files_limit': '54179', 'optimizer_prune_level': '3', 'optimizer_search_depth': '211', 'optimizer_trace_limit': '9', 'optimizer_trace_max_mem_size': '7163045', 'parser_max_mem_size': '143076243144818802688', 'password_history': '0', 'password_reuse_interval': '0', 'performance_schema_digests_size': '16322', 'performance_schema_error_size': '15315', 'performance_schema_events_stages_history_long_size': '29588', 'performance_schema_events_stages_history_size': '19', 'performance_schema_events_statements_history_long_size': '41942', 'performance_schema_events_statements_history_size': '52', 'performance_schema_events_transactions_history_long_size': '24564', 'performance_schema_events_transactions_history_size': '83', 'performance_schema_events_waits_history_long_size': '29376', 'performance_schema_events_waits_history_size': '97', 'performance_schema_max_cond_classes': '605', 'performance_schema_max_digest_length': '2739', 'performance_schema_max_digest_sample_age': '247', 'performance_schema_max_file_classes': '713', 'performance_schema_max_file_handles': '209268', 'performance_schema_max_memory_classes': '640', 'performance_schema_max_mutex_classes': '923', 'performance_schema_max_rwlock_classes': '117', 'performance_schema_max_socket_classes': '28', 'performance_schema_max_sql_text_length': '8116', 'performance_schema_max_stage_classes': '1237', 'performance_schema_max_statement_classes': '1251', 'performance_schema_max_statement_stack': '98', 'performance_schema_max_thread_classes': '187', 'performance_schema_session_connect_attrs_size': '2663', 'port': '8279', 'preload_buffer_size': '209586', 'profiling_history_size': '12', 'protocol_version': '18', 'query_alloc_block_size': '79744', 'query_prealloc_size': '2506', 'range_alloc_block_size': '40130', 'range_optimizer_max_mem_size': '21328818', 'read_buffer_size': '947271', 'read_rnd_buffer_size': '2454568', 'regexp_stack_limit': '32268884', 'regexp_time_limit': '102', 'relay_log_space_limit': '0', 'report_port': '32678', 'rpl_read_size': '10667', 'rpl_stop_slave_timeout': '194401766', 'schema_definition_cache': '2038', 'secondary_engine_cost_threshold': '346064', 'select_into_buffer_size': '674202', 'select_into_disk_sync_delay': '0', 'server_id': '2', 'server_id_bits': '188', 'slave_checkpoint_group': '1126', 'slave_checkpoint_period': '209', 'slave_max_allowed_packet': '6682193470', 'slave_net_timeout': '104', 'slave_parallel_workers': '0', 'slave_pending_jobs_size_max': '1292774253', 'slave_transaction_retries': '73', 'slow_launch_time': '7', 'sort_buffer_size': '2386885', 'sql_select_limit': '33818340816779427840', 'sql_slave_skip_counter': '0', 'stored_program_cache': '1148', 'stored_program_definition_cache': '1304', 'sync_binlog': '3', 'sync_master_info': '56582', 'sync_relay_log': '87477', 'sync_relay_log_info': '20524', 'table_definition_cache': '8002', 'table_open_cache': '7263', 'table_open_cache_instances': '59', 'tablespace_definition_cache': '1995', 'temptable_max_mmap': '8214479962', 'temptable_max_ram': '7465542271', 'thread_cache_size': '26', 'thread_stack': '1157554', 'tmp_table_size': '59866714', 'transaction_alloc_block_size': '64167', 'transaction_prealloc_size': '3445', 'wait_timeout': '221487'}
# # for param, val in good_conf.items():
    # # dbms.set_param(param, val)
# dbms.reconfigure()


# weighted_hints = {('innodb_buffer_pool_size', '0'): 4, ('innodb_buffer_pool_size', '5'): 12, ('innodb_buffer_pool_size', '4'): 12, ('innodb_buffer_pool_size', '80'): 4, ('innodb_buffer_pool_size', '57GB'): 4, ('innodb_buffer_pool_size', '1'): 8, ('innodb_buffer_pool_size', '256M'): 4, ('innodb_buffer_pool_size', '3'): 4, ('innodb_buffer_pool_size', '5G'): 4, ('innodb_buffer_pool_size', '10G'): 4, ('innodb_buffer_pool_size', '6000M'): 4}
# explorer = ParameterExplorer(None, None, None)
# print(explorer._select_configs(weighted_hints, 2))
#
#
# # Initialize unsupervised hint extraction environment
# dbms = MySQLconfig('tpch', 'root', 'mysql1234-', '/usr/local/mysql/bin')
# #dbms = PgConfig(db='tpch', user='immanueltrummer')
#
# success = dbms.set_param_smart('innodb_buffer_pool_size', '2GBMySQL')
# print(success)

# print(dbms.query_one('show global variables'))
#
# c = dbms.connection.cursor(buffered=True)
# c.execute('show global variables where variable_name != \'keyring_file_data\'')
# var_vals = c.fetchall()
# for var_val in var_vals:
    # var, val = var_val
    # success = dbms.set_param(var, 'default')
    # print(f'Success {success} for variable {var}')
# c.close()

#
# unlabeled_docs = DocCollection('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/postgres100', dbms)
# #unlabeled_docs = DocCollection('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/mysql100', dbms)
# tpc_c = TpcC('/Users/immanueltrummer/benchmarks/oltpbench', 
            # '/Users/immanueltrummer/benchmarks/oltpbench/config/tpcc_config_postgres.xml', 
            # '/Users/immanueltrummer/benchmarks/oltpbench/results', dbms, 'tpcctemplate', 'tpcc')
            #
# metrics = tpc_c.evaluate()
# print(metrics)
