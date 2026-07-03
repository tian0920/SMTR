from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    prediction_input_from_record,
)


class TransferFeatureLeakageScanner:
    forbidden = {
        "memory_id",
        "steps",
        "payload",
        "procedure_payload",
        "transfer_class",
        "y_share",
        "y_withhold",
        "team_reward",
        "scenario_family",
        "environment_regime",
        "prefix_structure_family",
        "factor_combination_id",
        "surface_variant_id",
        "mechanism_group_id",
        "branch_label",
        "forced_intervention",
    }

    def scan(self, records) -> dict:
        encoder = HashingTransferFeatureEncoder()
        violations = []
        for record in records:
            tokens = encoder.tokens(prediction_input_from_record(record))
            for token in tokens:
                for field in self.forbidden:
                    if field in token:
                        violations.append(
                            {
                                "record_id": record.record_id,
                                "offending_field": field,
                                "token_sample": token,
                            }
                        )
                        break
        return {"record_count": len(records), "violations": violations}
