from reglm.interpret import generate_random_sequences


def test_generate_random_sequences():
    gen = generate_random_sequences(n=10, seq_len=20)
    assert isinstance(gen, list)
    assert len(gen) == 10
    assert list(set([len(x) for x in gen])) == [20]
