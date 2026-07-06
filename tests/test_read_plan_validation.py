from rag_core.query_planner import _validated_llm_plan
from rag_core.multilingual import detect_language
from config import RagConfig

payload = {
  'intent': 'fact_lookup',
  'entity_mentions': [{'text': '艾达·洛夫莱斯', 'entity_type': 'person'}],
  'requested_property': '最终状态',
  'operation': 'lookup',
  'answer_mode': 'direct',
  'constraints': [],
  'answer_shape': 'concise conclusion',
  'read_strategy': {
     'scope': 'entity_neighborhood',
     'regions': ['terminal'],
     'position_bias': 'terminal',
     'need_timeline': True,
     'need_entity_neighborhood': True,
     'allow_partial': False,
  },
  'retrieval_queries': [{'text':'艾达·洛夫莱斯的最终状态','language':'zh'}],
  'planner_confidence': 0.92,
}
plan = _validated_llm_plan('艾达·洛夫莱斯的结局是什么？', detect_language('艾达·洛夫莱斯的结局是什么？'), payload, RagConfig())
assert plan is not None
assert plan.semantics.scope == 'entity_neighborhood'
assert plan.semantics.regions == ['terminal']
assert plan.semantics.need_entity_neighborhood is True
assert plan.preferred_chunk_kinds[0] == 'terminal_window'
print('read plan validation regression test passed')
