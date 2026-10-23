import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import certifi
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool


COMPANY_CONTEXT_QUERY = """
SELECT
    assessment_id,
    company_name,
    company_industry,
    company_hq_country,
    company_revenue,
    currency,
    company_employee_number,
    supply_chain_name,
    supply_chain_description,
    customer_regions,
    manufacturing_strategy,
    inhouse_manufacturing,
    outsourced_logistics,
    supply_chain_revenue,
    number_of_monthly_orders,
    number_of_direct_material_suppliers,
    number_of_manufacturing_plants,
    folder_id
FROM assessments__answers__demographic_assessment
WHERE folder_id = %s
LIMIT 1
"""

PROJECT_FOLDER_QUERY = """
SELECT folder_id
FROM account_management__projects
WHERE record_id = %s
LIMIT 1
"""


class CompanyContextDatabaseError(RuntimeError):
    pass


def _ssl_ca_path() -> str:
    configured = os.getenv("MYSQL_SSL_CA", "").strip()
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        if path.is_file():
            return str(path)
    return certifi.where()


def _required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise CompanyContextDatabaseError(f"Required database setting is missing: {name}")
    return value


@lru_cache(maxsize=1)
def _get_pool() -> MySQLConnectionPool:
    config: dict[str, Any] = {
        "host": _required_setting("MYSQL_HOST"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "database": _required_setting("MYSQL_DATABASE"),
        "user": _required_setting("MYSQL_USER"),
        "password": _required_setting("MYSQL_PASSWORD"),
        "charset": "utf8mb4",
        "collation": "utf8mb4_unicode_ci",
        "connection_timeout": int(os.getenv("MYSQL_CONNECTION_TIMEOUT", "10")),
        "ssl_disabled": os.getenv("MYSQL_SSL_DISABLED", "false").lower() == "true",
    }
    if not config["ssl_disabled"]:
        config["ssl_ca"] = _ssl_ca_path()
        config["ssl_verify_cert"] = True
    try:
        return MySQLConnectionPool(
            pool_name="company_context_pool",
            pool_size=int(os.getenv("MYSQL_POOL_SIZE", "5")),
            **config,
        )
    except (mysql.connector.Error, TypeError, ValueError) as exc:
        raise CompanyContextDatabaseError("Could not initialize the MySQL connection pool.") from exc


def get_company_context(folder_id: str) -> dict[str, Any] | None:
    normalized_id = str(folder_id).strip()
    if not normalized_id or len(normalized_id) > 128:
        raise ValueError("folder_id must be a non-empty value of at most 128 characters.")

    connection = None
    cursor = None
    try:
        connection = _get_pool().get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(COMPANY_CONTEXT_QUERY, (normalized_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return {key: value for key, value in row.items() if value is not None}
    except mysql.connector.Error as exc:
        raise CompanyContextDatabaseError("Could not read company context from MySQL.") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def get_project_folder_id(project_id: int) -> str | None:
    """Resolve the owning folder for a project using the project as authority."""
    try:
        normalized_id = int(project_id)
    except (TypeError, ValueError):
        raise ValueError("project_id must be a positive integer.") from None
    if normalized_id <= 0:
        raise ValueError("project_id must be a positive integer.")

    connection = None
    cursor = None
    try:
        connection = _get_pool().get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(PROJECT_FOLDER_QUERY, (normalized_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        folder_id = str(row.get("folder_id") or "").strip()
        return folder_id or None
    except mysql.connector.Error as exc:
        raise CompanyContextDatabaseError(
            "Could not resolve the project's folder from MySQL."
        ) from exc
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()
