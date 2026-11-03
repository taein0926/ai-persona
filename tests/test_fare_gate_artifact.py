from scripts.audit_private_result import station_fare_gate_ids


def test_serialized_gates_are_separate_from_stairs_and_exits():
    scenario={'stairs':[{'id':'A_W','kind':'stairs'},{'id':'E_1','kind':'stairs'}],
              'fare_gates':[{'id':'G_01'},{'id':'G_02'}]}
    assert station_fare_gate_ids(scenario)=={'G_01','G_02'}


def test_legacy_connector_format_is_supported_without_duplicates():
    scenario={'stairs':[{'id':'S','kind':'stairs'},{'id':'G','kind':'fare_gate'}],
              'fare_gates':[{'id':'G'},{'id':'SECOND_GATE'}]}
    assert station_fare_gate_ids(scenario)=={'G','SECOND_GATE'}
