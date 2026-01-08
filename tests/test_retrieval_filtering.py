from sqlalchemy.dialects import postgresql

import uuid

from app.services.retrieval_service import build_retrieval_query


def test_retrieval_filters_by_business_and_workspace():
    business_id = uuid.uuid4()
    workspace_id = uuid.uuid4()
    stmt = build_retrieval_query(
        business_id=business_id,
        workspace_id=workspace_id,
        query_embedding=[0.1, 0.2],
        top_k=5,
        similarity_threshold=0.0,
    )
    compiled = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert f"document_chunks.business_id = '{business_id}'" in compiled
    assert f"document_chunks.workspace_id = '{workspace_id}'" in compiled
