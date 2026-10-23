import re
from typing import Any

from .company_context_repository import CompanyContextDatabaseError, _get_pool


ALLOWED_SECTIONS = {
    "project", "processes", "process_profile", "process_io", "process_performance",
    "process_issues", "process_assessment", "organization", "steps", "definitions",
    "business_rules", "disconnects", "inputs_outputs", "metrics",
    "best_practices", "analysis_problems", "project_metrics", "project_best_practices",
}
PROCESS_SCOPED_SECTIONS = {
    "process_profile", "process_io", "process_performance", "process_issues",
    "process_assessment", "organization", "steps", "definitions", "business_rules", "disconnects",
    "inputs_outputs", "analysis_problems",
}


def _trim(value: Any, limit: int = 1200) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value


def _clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: _trim(value) for key, value in row.items() if value is not None}
        for row in rows
    ]


def _scope_process_rows(
    rows: list[dict[str, Any]], process_codes: set[str] | None
) -> list[dict[str, Any]]:
    cleaned = _clean_rows(rows)
    if not process_codes:
        return cleaned
    return [row for row in cleaned if str(row.get("code", "")).upper() in process_codes]


def _group_io_by_process(
    rows: list[dict[str, Any]]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        process_code = str(row.get("code", "")).strip()
        related_code = str(row.get("related_process_code", "")).strip()
        related_name = str(row.get("related_process_name", "")).strip()
        process_label = f"{related_code} | {related_name}" if related_name else related_code
        assessment = {
            field: value
            for field, value in row.items()
            if field not in {
                "code", "related_process_code", "related_process_name"
            }
        }
        grouped.setdefault(process_code, {}).setdefault(process_label, []).append(assessment)
    return grouped


def _group_process_details(
    rows: list[dict[str, Any]], detail_key: str, *, scope: str | None = None
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        process_code = str(row.get("code", ""))
        group = grouped.setdefault(
            process_code,
            {"process_code": process_code, detail_key: []},
        )
        if scope is not None:
            group["assignment_scope"] = scope
        if row.get("process_name"):
            group["process_name"] = row["process_name"]
        group[detail_key].append(
            {
                field: value
                for field, value in row.items()
                if field not in {"code", "process_name"}
            }
        )
    return list(grouped.values())


def _assemble_process_profiles(item: dict[str, Any]) -> list[dict[str, Any]]:
    rasci = {row["process_code"]: row for row in item.get("process_rasci", [])}
    execution = {
        row["process_code"]: row for row in item.get("process_execution", [])
    }

    def rows_by_code(key: str) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in item.get(key, []):
            code = str(row.get("code", ""))
            grouped.setdefault(code, []).append(
                {field: value for field, value in row.items() if field != "code"}
            )
        return grouped

    terms = rows_by_code("business_terms")
    rules = rows_by_code("business_rules")
    disconnects = rows_by_code("disconnects")
    profiles: list[dict[str, Any]] = []
    for process in item.get("selected_processes", []):
        code = str(process.get("code", ""))
        assignments = rasci.get(code, {}).get("assignments", [])
        responsible = [
            row for row in assignments if row.get("rasci_role") == "Responsible"
        ]
        accountable = [
            row for row in assignments if row.get("rasci_role") == "Accountable"
        ]
        profiles.append(
            {
                "process_code": code,
                "process_name": process.get("name"),
                "core_profile": {
                    "tools_used": process.get("process_tool"),
                    "process_frequency": process.get("process_frequency"),
                    "responsible": responsible,
                    "accountable": accountable,
                },
                "tools_used": process.get("process_tool"),
                "process_frequency": process.get("process_frequency"),
                "rasci_scope": "process",
                "rasci_assignments": assignments,
                "execution_steps": execution.get(code, {}).get("steps", []),
                "business_terms": terms.get(code, []),
                "business_rules": rules.get(code, []),
                "disconnects": disconnects.get(code, []),
            }
        )
    return profiles


def _assemble_process_io_profiles(item: dict[str, Any]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    input_groups = item.get("process_inputs_by_source_process", {})
    output_groups = item.get("process_outputs_by_consumer_process", {})
    for process in item.get("selected_processes", []):
        code = str(process.get("code", ""))
        profiles.append(
            {
                "process_code": code,
                "process_name": process.get("name"),
                "inputs": [
                    {"source_process": source, "assessments": assessments}
                    for source, assessments in input_groups.get(code, {}).items()
                ],
                "outputs": [
                    {"consumer_process": consumer, "assessments": assessments}
                    for consumer, assessments in output_groups.get(code, {}).items()
                ],
            }
        )
    return profiles


def _rows_by_process(
    rows: list[dict[str, Any]], process_field: str = "process_code"
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        code = str(row.get(process_field, ""))
        grouped.setdefault(code, []).append(
            {field: value for field, value in row.items() if field != process_field}
        )
    return grouped


def _assemble_process_performance_profiles(item: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = _rows_by_process(item.get("process_metrics", []))
    practices = _rows_by_process(item.get("process_best_practices", []))
    profiles: list[dict[str, Any]] = []
    for process in item.get("selected_processes", []):
        code = str(process.get("code", ""))
        process_metrics = metrics.get(code, [])
        measured_gaps = []
        missing_baselines = []
        for metric in process_metrics:
            current = metric.get("metric_value")
            target = metric.get("target_value")
            if current is None:
                missing_baselines.append(
                    {
                        "metric_code": metric.get("metric_code"),
                        "metric_name": metric.get("name"),
                        "assessment_status": metric.get("assessment_status"),
                        "target_value": target,
                        "unit": metric.get("unit"),
                    }
                )
            elif target is not None:
                measured_gaps.append(
                    {
                        "metric_code": metric.get("metric_code"),
                        "metric_name": metric.get("name"),
                        "current_value": current,
                        "target_value": target,
                        "unit": metric.get("unit"),
                    }
                )
        profiles.append(
            {
                "process_code": code,
                "process_name": process.get("name"),
                "associated_metrics": process_metrics,
                "associated_best_practices": practices.get(code, []),
                "measured_gaps": measured_gaps,
                "missing_baselines": missing_baselines,
            }
        )
    return profiles


def _assemble_process_issue_profiles(item: dict[str, Any]) -> list[dict[str, Any]]:
    disconnects = _rows_by_process(item.get("disconnects", []), "code")
    analyses = _rows_by_process(item.get("process_analysis", []), "code")
    problems = _rows_by_process(item.get("process_problems", []), "code")
    return [
        {
            "process_code": str(process.get("code", "")),
            "process_name": process.get("name"),
            "disconnects": disconnects.get(str(process.get("code", "")), []),
            "analyses": analyses.get(str(process.get("code", "")), []),
            "documented_problems": problems.get(str(process.get("code", "")), []),
        }
        for process in item.get("selected_processes", [])
    ]


def _evidence_tokens(value: Any) -> set[str]:
    ignored = {
        "and", "or", "the", "a", "an", "to", "of", "for", "in", "on", "with",
        "process", "product", "required", "existing", "defect", "assessment",
        "generation", "automated",
    }
    normalized: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", str(value or "").lower()):
        if len(token) <= 2 or token in ignored:
            continue
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        normalized.add(token)
    return normalized


def _extract_evidence_level(value: Any, label: str) -> str | None:
    match = re.search(
        rf"\b{re.escape(label)}\s*:\s*(High|Medium|Low)\b",
        str(value or ""),
        re.IGNORECASE,
    )
    return match.group(1).title() if match else None


def _build_priority_evidence(
    io_profile: dict[str, Any],
    performance_profile: dict[str, Any],
    issue_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    metrics = {
        str(metric.get("metric_code", "")): metric
        for metric in performance_profile.get("associated_metrics", [])
    }
    priorities: list[dict[str, Any]] = []
    documented = []
    for problem in issue_profile.get("documented_problems", []):
        priority = _extract_evidence_level(problem.get("problem_notes"), "Priority")
        affected_kpi = str(problem.get("affected_kpi", ""))
        metric_code = affected_kpi.split()[0] if affected_kpi else ""
        documented.append(
            {
                "evidence_type": "documented_problem",
                "priority": priority or "Unspecified",
                "title": problem.get("problem_name"),
                "description": problem.get("problem_description"),
                "owner_and_action": problem.get("problem_notes"),
                "affected_kpi": affected_kpi,
                "metric_evidence": metrics.get(metric_code),
            }
        )
    documented.sort(key=lambda row: severity_order.get(str(row["priority"]), 3))
    priorities.extend(documented)

    practices = performance_profile.get("associated_best_practices", [])
    evidence_candidates: list[tuple[int, dict[str, Any]]] = []
    io_groups = [
        ("input_defect", "source_process", group)
        for group in io_profile.get("inputs", [])
    ] + [
        ("output_defect", "consumer_process", group)
        for group in io_profile.get("outputs", [])
    ]
    for evidence_type, relation_field, group in io_groups:
        for assessment in group.get("assessments", []):
            if assessment.get("assessment_status") != "Defect":
                continue
            description = assessment.get("description")
            description_tokens = _evidence_tokens(description)
            supporting_practices = []
            for practice in practices:
                if len(description_tokens & _evidence_tokens(practice.get("name"))) >= 2:
                    supporting_practices.append(
                        {
                            "best_practice_code": practice.get("best_practice_code"),
                            "best_practice_name": practice.get("name"),
                            "maturity_rating": practice.get("maturity_rating"),
                            "recommended_action": practice.get("assessment_comments"),
                        }
                    )
            evidence_candidates.append(
                (
                    len(supporting_practices),
                    {
                        "evidence_type": evidence_type,
                        "priority": "Evidence-based",
                        "title": description,
                        relation_field: group.get(relation_field),
                        "status": "Defect",
                        "notes": assessment.get("notes"),
                        "supporting_best_practices": supporting_practices,
                    },
                )
            )
    evidence_candidates.sort(key=lambda row: (-row[0], str(row[1].get("title", ""))))
    priorities.extend(row for _, row in evidence_candidates)
    return [dict(row, evidence_rank=index) for index, row in enumerate(priorities, start=1)]


def _assemble_process_assessments(item: dict[str, Any]) -> list[dict[str, Any]]:
    collections = {
        "process_profile": _assemble_process_profiles(item),
        "input_output": _assemble_process_io_profiles(item),
        "performance": _assemble_process_performance_profiles(item),
        "issues": _assemble_process_issue_profiles(item),
    }
    indexed = {
        key: {row["process_code"]: row for row in rows}
        for key, rows in collections.items()
    }
    assessments: list[dict[str, Any]] = []
    for process in item.get("selected_processes", []):
        code = str(process.get("code", ""))
        profile = indexed["process_profile"].get(code, {})
        io_profile = indexed["input_output"].get(code, {})
        performance_profile = indexed["performance"].get(code, {})
        issue_profile = indexed["issues"].get(code, {})
        assessments.append(
            {
                "process_code": code,
                "process_name": process.get("name"),
                "process_profile": profile,
                "input_output": io_profile,
                "performance": performance_profile,
                "issues": issue_profile,
                "priority_evidence": _build_priority_evidence(
                    io_profile, performance_profile, issue_profile
                ),
            }
        )
    return assessments


def _read_project_metrics(cursor, project_id: int) -> dict[str, Any]:
    # Catalog relationships are intentionally absent: selection belongs to the project.
    cursor.execute(
        """SELECT s.metric_code, m.metric_name AS name, m.metric_unit AS unit,
                  a.metric_required, a.metric_exist, a.metric_disconnected,
                  a.metric_notes, a.frequency, a.tool, a.integration,
                  t.target_value
             FROM project__selected_metrics s
             LEFT JOIN scords_master__metrics m ON m.metric_code=s.metric_code
             LEFT JOIN project__metric_assessments a
               ON a.project_record_id=s.project_record_id AND a.metric_code=s.metric_code
             LEFT JOIN project__metric_targets t
               ON t.project_record_id=s.project_record_id AND t.metric_code=s.metric_code
            WHERE s.project_record_id=%s ORDER BY s.metric_code LIMIT 101""",
        (project_id,),
    )
    rows = cursor.fetchall()
    truncated = len(rows) > 100
    metrics = _clean_rows(rows[:100])
    values_truncated = False
    if metrics:
        codes = [row["metric_code"] for row in metrics]
        placeholders = ",".join(["%s"] * len(codes))
        cursor.execute(
            "SELECT metric_code, year, metric_value FROM project__metric_values "
            "WHERE project_record_id=%s AND metric_code IN (" + placeholders + ") "
            "ORDER BY metric_code, year, record_id LIMIT 2001",
            (project_id, *codes),
        )
        values = cursor.fetchall()
        values_truncated = len(values) > 2000
        grouped: dict[str, list[dict[str, Any]]] = {}
        for value in values[:2000]:
            grouped.setdefault(value["metric_code"], []).append(
                {"year": value["year"], "value": value["metric_value"]}
            )
        for metric in metrics:
            metric["values"] = grouped.get(metric["metric_code"], [])
            metric["assessment_status"] = (
                "Unassessed" if "metric_required" not in metric else
                "Not required" if not metric["metric_required"] else
                "Defect" if metric.get("metric_disconnected") else
                "Existing" if metric.get("metric_exist") else "Unassessed"
            )
    return {"scope": "project", "items": metrics, "truncated": truncated,
            "values_truncated": values_truncated, "limit": 100}


def _read_project_best_practices(cursor, project_id: int) -> dict[str, Any]:
    cursor.execute(
        """SELECT s.best_practice_code, m.best_practice_name AS name,
                  a.best_practice_assessed AS assessed, a.assessment_rating AS maturity_rating,
                  a.assessment_notes, a.assessment_comments
             FROM project__selected_best_practices s
             LEFT JOIN scords_master__best_practices m
               ON m.best_practice_code=s.best_practice_code
             LEFT JOIN project__best_practice_assessments a
               ON a.project_record_id=s.project_record_id
              AND a.best_practice_code=s.best_practice_code
            WHERE s.project_record_id=%s ORDER BY s.best_practice_code LIMIT 101""",
        (project_id,),
    )
    rows = cursor.fetchall()
    return {"scope": "project", "items": _clean_rows(rows[:100]),
            "truncated": len(rows) > 100, "limit": 100}


def get_current_project_data(
    folder_id: str,
    sections: list[str],
    project_id: int | None = None,
    process_codes: list[str] | None = None,
) -> dict[str, Any]:
    requested = list(dict.fromkeys(str(item).strip() for item in sections))
    invalid = set(requested) - ALLOWED_SECTIONS
    if not requested or invalid:
        raise ValueError(
            "sections must contain one or more of: project, processes, best_practices, metrics."
        )
    project_sections = {"project_metrics", "project_best_practices"} & set(requested)
    if project_sections and (project_id is None or process_codes):
        raise ValueError("Project-wide sections require project_id and no process-code filter.")
    if project_sections and set(requested) & PROCESS_SCOPED_SECTIONS:
        raise ValueError("Request project-wide and process-detail sections separately.")
    scoped_codes = {
        str(code).strip().upper() for code in (process_codes or []) if str(code).strip()
    } or None

    connection = None
    cursor = None
    try:
        connection = _get_pool().get_connection()
        cursor = connection.cursor(dictionary=True)
        project_filter = " AND record_id = %s" if project_id is not None else ""
        parameters: tuple[Any, ...] = (str(folder_id).strip(),)
        if project_id is not None:
            parameters += (int(project_id),)
        cursor.execute(
            """SELECT record_id, project_name, project_description, pe_bp_matching,
                      process_screener
                 FROM account_management__projects
                WHERE folder_id = %s AND is_active = 1"""
            + project_filter
            + " ORDER BY record_id DESC LIMIT 10",
            parameters,
        )
        projects = cursor.fetchall()
        if not projects:
            return {"found": False, "sections": requested, "projects": []}

        result_projects: list[dict[str, Any]] = []
        for project in projects:
            project_id = project["record_id"]
            item: dict[str, Any] = {"project_id": project_id}
            if "project" in requested:
                item["project"] = _clean_rows([project])[0]
            if "project_metrics" in requested:
                item["project_metrics"] = _read_project_metrics(cursor, project_id)
            if "project_best_practices" in requested:
                item["project_best_practices"] = _read_project_best_practices(cursor, project_id)

            if (
                "processes" in requested
                or set(requested)
                & {"process_profile", "process_io", "process_performance", "process_issues", "process_assessment"}
            ):
                cursor.execute(
                    """SELECT i.process_element_code AS code,
                              m.process_element_name AS name,
                              i.process_tool, i.process_frequency,
                              i.process_metrics_completed, i.process_analysis_completed,
                              i.process_problems_completed
                         FROM project__process_element_info i
                         LEFT JOIN scords_master__process_elements m
                           ON m.process_element_code = i.process_element_code
                        WHERE i.project_record_id = %s AND i.is_process_selected = 1
                        ORDER BY i.process_element_code LIMIT 200""",
                    (project_id,),
                )
                item["selected_processes"] = _scope_process_rows(cursor.fetchall(), scoped_codes)

            needs_process_scope = bool(set(requested) & PROCESS_SCOPED_SECTIONS)
            if needs_process_scope and scoped_codes is None:
                if "selected_processes" not in item:
                    cursor.execute(
                        """SELECT i.process_element_code AS code,
                                  m.process_element_name AS name
                             FROM project__process_element_info i
                             LEFT JOIN scords_master__process_elements m
                               ON m.process_element_code=i.process_element_code
                            WHERE i.project_record_id=%s AND i.is_process_selected=1
                            ORDER BY i.process_element_code LIMIT 200""",
                        (project_id,),
                    )
                    item["selected_processes"] = _clean_rows(cursor.fetchall())
                item["process_filter_required"] = True
                item["requested_detail_sections"] = [
                    section for section in requested if section in PROCESS_SCOPED_SECTIONS
                ]
                result_projects.append(item)
                continue

            if (
                "organization" in requested
                or "steps" in requested
                or set(requested) & {"process_profile", "process_assessment"}
            ):
                cursor.execute(
                    """SELECT process_element_code AS code,
                              (SELECT process_element_name FROM scords_master__process_elements
                                WHERE process_element_code=project__process_organization.process_element_code
                                LIMIT 1) AS process_name,
                              person_role AS rasci_role,
                              person_department AS department, person_title AS title,
                              person_name AS person
                         FROM project__process_organization
                        WHERE project_record_id=%s
                        ORDER BY process_element_code,
                          FIELD(person_role,'Responsible','Accountable','Supporting','Consulting','Informed')
                        LIMIT 500""",
                    (project_id,),
                )
                organization_rows = _scope_process_rows(cursor.fetchall(), scoped_codes)
                item["process_rasci"] = _group_process_details(
                    organization_rows, "assignments", scope="process"
                )

            if (
                "steps" in requested
                or "organization" in requested
                or set(requested) & {"process_profile", "process_assessment"}
            ):
                cursor.execute(
                    """SELECT process_element_code AS code,
                              (SELECT process_element_name FROM scords_master__process_elements
                                WHERE process_element_code=project__process_steps.process_element_code
                                LIMIT 1) AS process_name,
                              step_index, step_description, step_notes
                         FROM project__process_steps WHERE project_record_id=%s
                        ORDER BY process_element_code, CAST(step_index AS UNSIGNED), record_id LIMIT 500""",
                    (project_id,),
                )
                step_rows = _scope_process_rows(cursor.fetchall(), scoped_codes)
                item["process_execution"] = _group_process_details(step_rows, "steps")

            if "definitions" in requested or set(requested) & {"process_profile", "process_assessment"}:
                cursor.execute(
                    """SELECT process_element_code AS code, definition_index,
                              definition_name, definition_description
                         FROM project__process_definitions WHERE project_record_id=%s
                        ORDER BY process_element_code, CAST(definition_index AS UNSIGNED), record_id LIMIT 500""",
                    (project_id,),
                )
                item["business_terms"] = _scope_process_rows(cursor.fetchall(), scoped_codes)

            if "business_rules" in requested or set(requested) & {"process_profile", "process_assessment"}:
                cursor.execute(
                    """SELECT process_element_code AS code, business_rule_index,
                              business_rule_name, business_rule_description
                         FROM project__process_business_rules WHERE project_record_id=%s
                        ORDER BY process_element_code, CAST(business_rule_index AS UNSIGNED), record_id LIMIT 500""",
                    (project_id,),
                )
                item["business_rules"] = _scope_process_rows(cursor.fetchall(), scoped_codes)

            if "disconnects" in requested or set(requested) & {"process_profile", "process_issues", "process_assessment"}:
                cursor.execute(
                    """SELECT process_element_code AS code, disconnect_index,
                              disconnect_name, disconnect_description, disconnect_notes
                         FROM project__process_disconnects WHERE project_record_id=%s
                        ORDER BY process_element_code, CAST(disconnect_index AS UNSIGNED), record_id LIMIT 500""",
                    (project_id,),
                )
                item["disconnects"] = _scope_process_rows(cursor.fetchall(), scoped_codes)

            if "inputs_outputs" in requested or set(requested) & {"process_io", "process_assessment"}:
                cursor.execute(
                    """SELECT r.record_id AS relation_id, r.process_element_code AS code,
                              r.process_element_input AS related_process_code,
                              pm.process_element_name AS related_process_name,
                              r.process_element_input_description AS description,
                              a.process_input_required AS required,
                              CASE
                                WHEN a.process_input_required=0 THEN 'Not required'
                                WHEN a.process_input_disconnected=1 THEN 'Defect'
                                WHEN a.process_input_exist=1 THEN 'Existing'
                                ELSE 'Unassessed'
                              END AS assessment_status,
                              a.process_input_notes AS notes,
                              a.communication, a.tool, a.integration
                         FROM project__process_input_assessments a
                         JOIN scords_relations__process_element_inputs r
                           ON r.record_id=a.process_element_input_record_id
                         LEFT JOIN scords_master__process_elements pm
                           ON pm.process_element_code=r.process_element_input
                        WHERE a.project_record_id=%s ORDER BY r.process_element_code,r.record_id LIMIT 500""",
                    (project_id,),
                )
                input_rows = _scope_process_rows(cursor.fetchall(), scoped_codes)
                item["process_inputs_by_source_process"] = _group_io_by_process(input_rows)
                cursor.execute(
                    """SELECT r.record_id AS relation_id, r.process_element_code AS code,
                              r.process_element_output AS related_process_code,
                              pm.process_element_name AS related_process_name,
                              r.process_element_output_description AS description,
                              a.process_output_required AS required,
                              CASE
                                WHEN a.process_output_required=0 THEN 'Not required'
                                WHEN a.process_output_disconnected=1 THEN 'Defect'
                                WHEN a.process_output_exists=1 THEN 'Existing'
                                ELSE 'Unassessed'
                              END AS assessment_status,
                              a.process_output_notes AS notes,
                              a.communication, a.tool, a.integration
                         FROM project__process_output_assessments a
                         JOIN scords_relations__process_element_outputs r
                           ON r.record_id=a.process_element_output_record_id
                         LEFT JOIN scords_master__process_elements pm
                           ON pm.process_element_code=r.process_element_output
                        WHERE a.project_record_id=%s ORDER BY r.process_element_code,r.record_id LIMIT 500""",
                    (project_id,),
                )
                output_rows = _scope_process_rows(cursor.fetchall(), scoped_codes)
                item["process_outputs_by_consumer_process"] = _group_io_by_process(output_rows)

            if "best_practices" in requested or set(requested) & {"process_performance", "process_assessment"}:
                relationship_filter = ""
                relationship_parameters: tuple[Any, ...] = (project_id,)
                if scoped_codes:
                    ordered_codes = sorted(scoped_codes)
                    relationship_filter = " AND r.process_element_code IN (" + ",".join(
                        ["%s"] * len(ordered_codes)
                    ) + ")"
                    relationship_parameters += tuple(ordered_codes)
                cursor.execute(
                    """SELECT r.process_element_code AS process_code,
                              s.best_practice_code AS best_practice_code,
                              m.best_practice_name AS name,
                              a.best_practice_assessed AS assessed,
                              a.assessment_rating AS maturity_rating,
                              a.assessment_notes, a.assessment_comments
                         FROM project__selected_best_practices s
                         JOIN scords_relations__process_element_best_practices r
                           ON r.best_practice_code=s.best_practice_code
                         LEFT JOIN scords_master__best_practices m
                           ON m.best_practice_code = s.best_practice_code
                         LEFT JOIN project__best_practice_assessments a
                           ON a.project_record_id = s.project_record_id
                          AND a.best_practice_code = s.best_practice_code
                        WHERE s.project_record_id = %s"""
                    + relationship_filter
                    + " ORDER BY s.best_practice_code LIMIT 500",
                    relationship_parameters,
                )
                practice_rows = _clean_rows(cursor.fetchall())
                if scoped_codes:
                    practice_rows = [
                        row for row in practice_rows
                        if str(row.get("process_code", "")).upper() in scoped_codes
                    ]
                item["process_best_practices"] = practice_rows
                if "best_practices" in requested:
                    item["best_practices"] = practice_rows

            if "metrics" in requested or set(requested) & {"process_performance", "process_assessment"}:
                relationship_filter = ""
                relationship_parameters = (project_id,)
                if scoped_codes:
                    ordered_codes = sorted(scoped_codes)
                    relationship_filter = " AND r.process_element_code IN (" + ",".join(
                        ["%s"] * len(ordered_codes)
                    ) + ")"
                    relationship_parameters += tuple(ordered_codes)
                cursor.execute(
                    """SELECT r.process_element_code AS process_code,
                              s.metric_code AS metric_code, m.metric_name AS name,
                              m.metric_unit AS unit, a.metric_required,
                              CASE
                                WHEN a.metric_required=0 THEN 'Not required'
                                WHEN a.metric_disconnected=1 THEN 'Defect'
                                WHEN a.metric_exist=1 THEN 'Existing'
                                ELSE 'Unassessed'
                              END AS assessment_status,
                              a.frequency, a.tool, a.integration,
                              v.metric_value, v.year, t.target_value
                         FROM project__selected_metrics s
                         JOIN scords_relations__process_element_metrics r
                           ON r.metric_code=s.metric_code
                         LEFT JOIN scords_master__metrics m ON m.metric_code = s.metric_code
                         LEFT JOIN project__metric_assessments a
                           ON a.project_record_id = s.project_record_id AND a.metric_code = s.metric_code
                         LEFT JOIN project__metric_values v
                           ON v.project_record_id = s.project_record_id AND v.metric_code = s.metric_code
                         LEFT JOIN project__metric_targets t
                           ON t.project_record_id = s.project_record_id AND t.metric_code = s.metric_code
                        WHERE s.project_record_id = %s"""
                    + relationship_filter
                    + " ORDER BY s.metric_code, v.year LIMIT 500",
                    relationship_parameters,
                )
                metric_rows = _clean_rows(cursor.fetchall())
                if scoped_codes:
                    metric_rows = [
                        row for row in metric_rows
                        if str(row.get("process_code", "")).upper() in scoped_codes
                    ]
                item["process_metrics"] = metric_rows
                if "metrics" in requested:
                    item["metrics"] = metric_rows
            if "analysis_problems" in requested or set(requested) & {"process_issues", "process_assessment"}:
                cursor.execute(
                    """SELECT process_element_code AS code, analysis_index, analysis_name,
                              analysis_description, potential_problem
                         FROM project__process_analysis WHERE project_record_id=%s
                        ORDER BY process_element_code, CAST(analysis_index AS UNSIGNED), record_id LIMIT 500""",
                    (project_id,),
                )
                item["process_analysis"] = _scope_process_rows(cursor.fetchall(), scoped_codes)
                cursor.execute(
                    """SELECT process_element_code AS code, problem_name, problem_description,
                              problem_notes, affected_kpi
                         FROM project__process_problems WHERE project_record_id=%s
                        ORDER BY process_element_code, record_id LIMIT 500""",
                    (project_id,),
                )
                item["process_problems"] = _scope_process_rows(cursor.fetchall(), scoped_codes)
            if "process_profile" in requested:
                item = {
                    "project_id": project_id,
                    "process_profiles": _assemble_process_profiles(item),
                }
            elif "process_io" in requested:
                item = {
                    "project_id": project_id,
                    "process_io_profiles": _assemble_process_io_profiles(item),
                }
            elif "process_performance" in requested:
                item = {
                    "project_id": project_id,
                    "process_performance_profiles": _assemble_process_performance_profiles(item),
                }
            elif "process_issues" in requested:
                item = {
                    "project_id": project_id,
                    "process_issue_profiles": _assemble_process_issue_profiles(item),
                }
            elif "process_assessment" in requested:
                item = {
                    "project_id": project_id,
                    "process_assessments": _assemble_process_assessments(item),
                }
            result_projects.append(item)
        return {"found": True, "sections": requested, "projects": result_projects}
    except Exception as exc:
        if isinstance(exc, (ValueError, CompanyContextDatabaseError)):
            raise
        raise CompanyContextDatabaseError("Could not read current project data from MySQL.") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()
