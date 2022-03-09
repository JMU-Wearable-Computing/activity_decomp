from skillest.dataloaders import UIPRMDDataloader, UIPRMDSingleDataloader
from skillest.dataloaders.ui_prmd_dl import get_data
import torch
import pytest

single_dl = UIPRMDSingleDataloader(batch_size=-1, num_ep_in_train=6, num_ep_in_val=2, num_ep_in_test=2)
dl = UIPRMDSingleDataloader(batch_size=-1, num_ep_in_train=6, num_ep_in_val=2, num_ep_in_test=2)

def test_seed():
    dl2 = UIPRMDSingleDataloader(batch_size=-1, num_ep_in_train=6, num_ep_in_val=2, num_ep_in_test=2)
    tvalues, tlabels, tmovement, tsubject = dl.sample_data(dl.sub_train, dl.mov_train, dl.ep_train)
    tvalues2, tlabels2, tmovement2, tsubject2 = dl2.sample_data(dl2.sub_train, dl2.mov_train, dl2.ep_train)

    assert torch.equal(tvalues, tvalues2)
    assert torch.equal(tlabels, tlabels2)
    assert torch.equal(tmovement, tmovement2)
    assert torch.equal(tsubject, tsubject2)


def test_correct_separation():

    single_dl = UIPRMDSingleDataloader(batch_size=-1, num_ep_in_train=6, num_ep_in_val=2, num_ep_in_test=2)
    tvalues, tlabels, tmovement, tsubject = single_dl.sample_data(single_dl.sub_train, single_dl.mov_train, single_dl.ep_train)
    vvalues, vlabels, vmovement, vsubject = single_dl.sample_data(single_dl.sub_val, single_dl.mov_val, single_dl.ep_val)

    for i in range(1, 11):
        for j in range(1, 11):
            single_dl.set_subject(subject=i)
            single_dl.set_movement(movement=j)
            dt = iter(single_dl.train_dataloader())
            dv = iter(single_dl.val_dataloader())

            tmask = (tsubject == i) & (tmovement == j)
            vmask = (vsubject == i) & (vmovement == j)

            tv, tl = next(dt)
            vv, vl = next(dv)

            # print(tvalues[tmask, ...].shape)
            print(tv.shape)
            assert torch.equal(torch.sort(tv, axis=0)[0], torch.sort(tvalues[tmask, ...], axis=0)[0])
            assert torch.equal(torch.sort(vv, axis=0)[0], torch.sort(vvalues[vmask, ...], axis=0)[0])

            assert torch.equal(torch.sort(tl, axis=0)[0], torch.sort(tlabels[tmask, ...], axis=0)[0])
            assert torch.equal(torch.sort(vl, axis=0)[0], torch.sort(vlabels[vmask, ...], axis=0)[0])



