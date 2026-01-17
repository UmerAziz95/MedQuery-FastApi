from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session

router = APIRouter()


@router.get("/db/overview")
async def db_overview(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    tables_result = await session.execute(
        text(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        )
    )
    tables = [row[0] for row in tables_result.fetchall()]

    table_entries = []
    for table_name in tables:
        count_result = await session.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
        table_entries.append(
            {
                "name": table_name,
                "row_count": int(count_result.scalar_one()),
            }
        )

    meta_result = await session.execute(text("SELECT current_database(), current_user, version()"))
    database_name, database_user, database_version = meta_result.one()

    size_result = await session.execute(text("SELECT pg_database_size(current_database())"))
    database_size_bytes = size_result.scalar_one()

    return {
        "meta": {
            "name": database_name,
            "user": database_user,
            "version": database_version,
            "size_bytes": int(database_size_bytes),
            "schema": "public",
        },
        "tables": table_entries,
        "table_count": len(table_entries),
    }


@router.get("/db/tables/{table_name}")
async def db_table_detail(
    table_name: str, session: AsyncSession = Depends(get_session)
) -> dict[str, Any]:
    table_exists_result = await session.execute(
        text(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
              AND table_name = :table_name
            """
        ),
        {"table_name": table_name},
    )

    if table_exists_result.scalar() is None:
        raise HTTPException(status_code=404, detail="Table not found")

    count_result = await session.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
    row_count = int(count_result.scalar_one())

    columns_result = await session.execute(
        text(
            """
            SELECT column_name, data_type, is_nullable, column_default, ordinal_position
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = :table_name
            ORDER BY ordinal_position
            """
        ),
        {"table_name": table_name},
    )
    columns = [
        {
            "name": row[0],
            "data_type": row[1],
            "is_nullable": row[2],
            "default": row[3],
            "position": int(row[4]),
        }
        for row in columns_result.fetchall()
    ]

    return {
        "name": table_name,
        "row_count": row_count,
        "columns": columns,
    }
