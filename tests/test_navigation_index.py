from pathlib import Path
from rag_core.cards import analyze_document, build_cards, build_document_map
from rag_core.hybrid_index import HybridIndex, HybridRecord

pages = []
for page in range(1, 13):
    if page < 10:
        text = f"第 {page} 页\n\n艾达·洛夫莱斯在前期调查事件 {page}。"
    elif page == 10:
        text = "第 10 页\n\n艾达·洛夫莱斯完成了任务，并与同伴告别。"
    elif page == 11:
        text = "第 11 页\n\n艾达·洛夫莱斯离开城市，故事的主要冲突已经结束。"
    else:
        text = "第 12 页\n\n尾声：艾达·洛夫莱斯开始新的生活。"
    pages.append({"text": text, "metadata": {"page": page, "page_count": 12, "page_block_index": 0, "source_type": "pdf"}})

signals = analyze_document('示例作品.pdf', pages)
doc_map = build_document_map('示例作品.pdf', pages, signals)
assert doc_map.total_units == 12
assert doc_map.metadata_for_unit(11)['navigation_region'] == 'terminal'
cards = build_cards('示例作品.pdf','x.pdf','doc-1','now','hash',pages, doc_map)
assert any(card.metadata['chunk_kind'] == 'terminal_window' for card in cards)

index_path = Path('/tmp/navigable_rag_test/index.sqlite3')
if index_path.exists(): index_path.unlink()
idx = HybridIndex(path=index_path, collection_name='demo')
records=[]
for i, card in enumerate(cards):
    metadata=dict(card.metadata)
    metadata['chunk_id']=f'card-{i}'
    records.append(HybridRecord(
       chunk_id=f'card-{i}', document_id='doc-1', parent_chunk_id='',
       chunk_kind=metadata['chunk_kind'], language='zh', title='示例作品',
       section_path=str(metadata.get('section_path','')), content=card.content,
       metadata=metadata, aliases=list(metadata.get('aliases') or []), exact_terms=[]))
for i, page in enumerate(pages):
    metadata={**page['metadata'], **doc_map.metadata_for_unit(i), 'chunk_id':f'body-{i}',
              'document_id':'doc-1','document_title':'示例作品','chunk_kind':'child',
              'page_start':i+1,'page_end':i+1}
    records.append(HybridRecord(
       chunk_id=f'body-{i}', document_id='doc-1', parent_chunk_id='', chunk_kind='child',
       language='zh', title='示例作品', section_path=str(metadata.get('section_path','')),
       content=page['text'], metadata=metadata, aliases=list(metadata.get('aliases') or []), exact_terms=[]))
idx.upsert_records(records)
nav = idx.navigation_search(['doc-1'], ['terminal'], query='艾达·洛夫莱斯的结局是什么', language='zh', position_bias='terminal')
assert nav, 'terminal navigation should return raw source windows'
assert any(hit.metadata.get('navigation_region') == 'terminal' for hit in nav)
entity = idx.entity_terminal_neighborhood_search(['艾达洛夫莱斯'], ['doc-1'])
assert entity, 'entity terminal neighbourhood should resolve indexed aliases'
assert any('尾声' in hit.content or '主要冲突已经结束' in hit.content for hit in entity)
print('navigation index regression test passed:', len(cards), len(nav), len(entity))
