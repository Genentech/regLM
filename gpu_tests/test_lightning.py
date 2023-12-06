import torch

from reglm.lightning import LightningModel

logits = torch.Tensor(
    [
        [
            [-0.2471, -0.0181, -0.0437, 0.0712, 1.7542],
            [0.1820, 0.1408, 0.0537, 0.2158, -0.0734],
            [1.0527, 0.1054, -0.0512, 0.0798, 0.3800],
            [0.0793, 0.2634, 0.3122, 0.4130, -0.0483],
            [0.1415, 0.2421, 0.3021, 0.3576, 0.3998],
            [-0.4842, -0.3171, -0.2690, -0.1472, -0.0694],
            [0.5184, 0.3948, 0.3564, 0.2577, -0.4265],
            [-0.1820, 0.9578, 0.9522, 1.0789, -0.2434],
            [0.8964, 0.8879, 0.9174, 0.7806, 0.0500],
            [0.4533, -0.5569, -0.3792, -0.3816, 0.0417],
            [-0.2447, -0.2324, -0.2306, -0.2032, -0.3107],
            [0.1941, -0.4652, -0.7622, -0.4444, 0.0758],
            [0.3891, -0.2421, -0.4224, -0.3631, 0.1349],
            [0.0211, -0.3550, -0.5738, -0.4727, -0.0234],
            [0.1715, -0.0564, -0.2032, -0.2841, 0.1306],
            [-0.2471, -0.2780, -0.2100, -0.3440, 0.0125],
        ]
    ]
)

probs = torch.Tensor(
    [
        [
            [0.0377, 0.0538, 0.0539, 0.0579, 0.2731],
            [0.0580, 0.0631, 0.0594, 0.0669, 0.0439],
            [0.1384, 0.0609, 0.0535, 0.0584, 0.0691],
            [0.0523, 0.0713, 0.0769, 0.0815, 0.0450],
            [0.0557, 0.0698, 0.0762, 0.0771, 0.0705],
            [0.0298, 0.0399, 0.0430, 0.0466, 0.0441],
            [0.0811, 0.0813, 0.0804, 0.0698, 0.0308],
            [0.0403, 0.1428, 0.1459, 0.1586, 0.0370],
            [0.1184, 0.1332, 0.1409, 0.1177, 0.0497],
            [0.0760, 0.0314, 0.0385, 0.0368, 0.0493],
            [0.0378, 0.0434, 0.0447, 0.0440, 0.0346],
            [0.0587, 0.0344, 0.0263, 0.0346, 0.0510],
            [0.0713, 0.0430, 0.0369, 0.0375, 0.0541],
            [0.0493, 0.0384, 0.0317, 0.0336, 0.0462],
            [0.0574, 0.0518, 0.0460, 0.0406, 0.0538],
            [0.0377, 0.0415, 0.0456, 0.0382, 0.0478],
        ]
    ]
)

lik = torch.Tensor([0.1384, 0.0713, 0.1459, 0.1177, 0.0493])

config = {
    "d_model": 256,
    "n_layer": 8,
    "d_inner": 1024,
    "vocab_size": 12,
    "resid_dropout": 0.0,
    "embed_dropout": 0.1,
    "fused_mlp": False,
    "fused_dropout_add_ln": True,
    "residual_in_fp32": True,
    "pad_vocab_size_multiple": 8,
    "return_hidden_state": True,
    "layer": {
        "emb_dim": 5,
        "filter_order": 64,
        "local_order": 3,
        "l_max": 88,
        "modulate": True,
        "w": 10,
        "lr": 0.0006,
        "wd": 0.0,
        "lr_pos_emb": 0.0,
        "_name_": "hyena",
    },
}


def test_lightningmodel():
    # Build model
    model = LightningModel(config=config, logger=None, label_len=2)
    model.seq_len = 4

    # Encoding
    assert torch.all(model.encode_labels("10") == torch.LongTensor([3, 2]))
    assert torch.all(model.encode_seqs("ACGT") == torch.LongTensor([7, 8, 9, 10]))
    assert torch.all(
        model.encode_seqs("ACGT", add_stop=True) == torch.LongTensor([7, 8, 9, 10, 1])
    )
    assert torch.all(
        model.encode("ACGT", "01") == torch.LongTensor([2, 3, 7, 8, 9, 10])
    )
    assert torch.all(
        model.encode("ACGT", "01", add_stop=True)
        == torch.LongTensor([2, 3, 7, 8, 9, 10, 1])
    )

    # Decoding
    assert model.decode(torch.LongTensor([7, 8, 9, 10])) == ["ACGT"]

    # Probabilities
    assert torch.allclose(model.logits_to_probs(logits), probs, rtol=1e-2)
    assert torch.allclose(
        model.probs_to_likelihood(torch.LongTensor([2, 3, 7, 8, 9]), probs), lik
    )
    fprobs = torch.Tensor([[0.14788991, 0.43449541, 0.27889908, 0.1387156]])
    assert torch.allclose(
        model.filter_base_probs(probs[:, :, 0], normalize=True), fprobs
    )
    assert torch.allclose(
        model.threshold_probs(fprobs, top_p=0.7),
        torch.Tensor([[0.0, 0.4345, 0.2789, 0.0]]),
        rtol=1e-2,
    )
    fprobs = torch.Tensor([[0.14788991, 0.43449541, 0.27889908, 0.1387156]])
    assert torch.allclose(
        model.threshold_probs(fprobs, top_k=1),
        torch.Tensor([[0.0, 0.4345, 0.0, 0.0]]),
        rtol=1e-2,
    )

    # Likelihoods
    model.probs_to_likelihood

    # Generation
    rng = torch.Generator()
    rng.manual_seed(0)
    assert model.logits_to_idxs(logits[:, :, 0], random_state=rng) == torch.tensor(9)
