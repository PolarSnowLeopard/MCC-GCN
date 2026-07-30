"""Pair-level assessment of neutral-reactant candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .standardize import (
    ComponentStandardization,
    standardize_exported_component,
)


@dataclass(frozen=True)
class PairNeutralizationAssessment:
    status: str
    observed_charge_sum: int | None
    candidate_charge_sum: int | None
    hydrogen_delta_sum: int | None
    proton_transfer_consistent: bool | None
    warnings: tuple[str, ...]
    component_A: ComponentStandardization
    component_B: ComponentStandardization

    def to_dict(self):
        return asdict(self)


def _sum_optional(first: int | None, second: int | None) -> int | None:
    if first is None or second is None:
        return None
    return first + second


def assess_pair_candidates(
    component_a: dict,
    component_b: dict,
) -> PairNeutralizationAssessment:
    """Assess conservation checks without approving either candidate."""
    return assess_standardized_pair(
        standardize_exported_component(component_a),
        standardize_exported_component(component_b),
    )


def assess_standardized_pair(
    result_a: ComponentStandardization,
    result_b: ComponentStandardization,
) -> PairNeutralizationAssessment:
    """Assess a pair of cached component-standardization results."""
    components = (result_a, result_b)

    observed_charge_sum = _sum_optional(
        result_a.observed_formal_charge,
        result_b.observed_formal_charge,
    )
    candidate_charge_sum = _sum_optional(
        result_a.candidate_formal_charge,
        result_b.candidate_formal_charge,
    )
    hydrogen_delta_sum = _sum_optional(
        result_a.hydrogen_delta,
        result_b.hydrogen_delta,
    )
    if any(
        result.observed_formal_charge is None
        or result.hydrogen_delta is None
        for result in components
    ):
        proton_transfer_consistent = None
    else:
        proton_transfer_consistent = all(
            result.hydrogen_delta == -result.observed_formal_charge
            for result in components
        )
    warnings = tuple(
        f"component_{side}:{warning}"
        for side, result in zip(("A", "B"), components)
        for warning in result.warnings
    )

    if any(result.error for result in components):
        status = "component_parse_error"
    elif any(
        result.candidate_status == "candidate_rejected"
        for result in components
    ):
        status = "component_candidate_rejected"
    elif all(
        result.candidate_status == "observed_neutral"
        for result in components
    ):
        status = "observed_neutral_pair"
    elif any(
        result.candidate_status == "unresolved_charged"
        for result in components
    ):
        status = "unresolved_charged_component"
    elif observed_charge_sum != 0:
        status = "stoichiometry_required"
    elif candidate_charge_sum != 0:
        status = "candidate_charge_imbalance"
    elif hydrogen_delta_sum != 0 or not proton_transfer_consistent:
        status = "candidate_hydrogen_imbalance"
    elif any(
        result.local_candidate_status == "ambiguous_candidates"
        for result in components
        if result.observed_formal_charge
    ):
        status = "local_neutralization_ambiguous"
    elif any(
        result.local_candidate_status
        in {"configuration_limit_exceeded", "no_valid_candidate"}
        for result in components
        if result.observed_formal_charge
    ):
        status = "local_neutralization_unresolved"
    elif any(
        result.charge_parent_matches_local_candidate is not True
        for result in components
        if result.observed_formal_charge
    ):
        status = "charge_parent_local_mismatch"
    else:
        observed_charges = sorted(
            (
                result_a.observed_formal_charge,
                result_b.observed_formal_charge,
            )
        )
        hydrogen_deltas = sorted(
            (result_a.hydrogen_delta, result_b.hydrogen_delta)
        )
        if observed_charges == [-1, 1] and hydrogen_deltas == [-1, 1]:
            status = "balanced_monovalent_candidate"
        else:
            status = "balanced_multivalent_candidate"

    return PairNeutralizationAssessment(
        status=status,
        observed_charge_sum=observed_charge_sum,
        candidate_charge_sum=candidate_charge_sum,
        hydrogen_delta_sum=hydrogen_delta_sum,
        proton_transfer_consistent=proton_transfer_consistent,
        warnings=warnings,
        component_A=result_a,
        component_B=result_b,
    )
