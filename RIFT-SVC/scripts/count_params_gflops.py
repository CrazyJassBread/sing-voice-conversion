import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rift_svc import RF, DiT
import torch
import thop
import click


@click.command()
@click.option('--dim', type=int, default=768, help='Transformer dimension')
@click.option('--depth', type=int, default=12, help='Transformer depth')
@click.option('--ff_mult', type=int, default=4, help='Feedforward multiplier')
@click.option('--n_mel_channels', type=int, default=128, help='Number of mel channels')
@click.option('--frame_len', type=int, default=1024, help='Frame length')
def main(dim, depth, ff_mult, n_mel_channels, frame_len):
    cvec_dim = 768
    transformer = DiT(
        dim=dim,
        depth=depth,
        head_dim=64,
        ff_mult=ff_mult,
        cvec_dim=cvec_dim
    )

    model = RF(transformer=transformer)

    flops, params = thop.profile(
        model,
        inputs=(
            torch.randn(1, frame_len, n_mel_channels),  # mel
            torch.full((1,), 0),  # spk_id
            torch.randn(1, frame_len),  # f0
            torch.randn(1, frame_len),  # rms
            torch.randn(1, frame_len, cvec_dim)
        )
    )

    print(f"dim: {dim}, depth: {depth}")
    print(f"FLOPs: {flops / 1e9} G")
    print(f"Params: {params / 1e6} M")

if __name__ == '__main__':
    main()
