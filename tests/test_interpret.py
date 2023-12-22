from reglm.interpret import ISM, generate_random_sequences


def test_generate_random_sequences():
    gen = generate_random_sequences(n=10, seq_len=20)
    assert isinstance(gen, list)
    assert len(gen) == 10
    assert list(set([len(x) for x in gen])) == [20]


def test_ISM():
    ism_output = {
        "AT",
        "CT",
        "GT",
        "TT",
        "AA",
        "AC",
        "AG",
        "AT",
    }
    assert set(ISM("AT", drop_ref=False)) == ism_output
    ism_output = {
        "CT",
        "GT",
        "TT",
        "AA",
        "AC",
        "AG",
    }
    assert set(ISM("AT", drop_ref=True)) == ism_output
