DB-BERT extracts tuning hints for database systems from text documents such as the manual. It uses extracted hints as a starting point for automated performance tuning. 

To analyze text, DB-BERT uses pre-trained language models such as BERT. During tuning, DB-BERT iteratively selects parameter settings and measures performance on a user-defined benchmark. It selects settings to try via reinforcement learning, integrating information extracted from text as well as performance measurements in past iterations. By exploiting text as additional input, DB-BERT tends to find promising configurations faster than baselines.

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/JFQbrK5GgFk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

# Publications

- **SIGMOD Record 2021** Database tuning using natural language processing. _Immanuel Trummer_.
- **VLDB 2021** The case for nlp-enhanced database tuning: towards tuning tools that "read the manual". _Immanuel Trummer_.
