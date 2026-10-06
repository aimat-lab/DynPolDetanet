"""Factory functions that reconstruct a DetaNet model with the exact
hyperparameters used to train each released checkpoint, then load its
weights from `trained_param/`.

Why this exists: `DetaNet.__init__` needs ~15 hyperparameters (feature
size, number of interaction blocks, output type, etc.) to be reconstructed
identically to how a given checkpoint was trained -- `load_state_dict`
will fail (or silently mismatch) if any of them differ. Rather than
requiring callers to remember/hardcode those values themselves, each
function below pins them to whatever was actually used for that specific
checkpoint, so loading a pretrained model is just e.g.
`model_loader.hopv241_dynamic_polarizability_model(device=device)`.

For the two models relevant to the paper (`qm9s_dynamic_polarizability_model`
and `hopv241_dynamic_polarizability_model`, near the bottom of this file):
see evaluate.py for a
worked example of loading and evaluating them.

Checkpoint paths default to being relative to wherever the calling script
is run from (typically the `code/` directory); pass an absolute path via
`params=` if running from elsewhere.
"""
import torch
from .detanet import DetaNet
def scalar_model(device,params='trained_param/qm7x/energy.pth',max_number=9):
    '''Loads the QM7-X-trained scalar energy-prediction model.'''
    state_dict = torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=max_number,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=True,
                    norm=False,
                    out_type='scalar',
                    grad_type=None,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def force_model(device,params='trained_param/qm7x/force.pth'):
    '''Loads the QM7-X-trained atomic-force model (energy gradient w.r.t. positions).'''
    state_dict = torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=17,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=True,
                    norm=False,
                    out_type='scalar',
                    grad_type='force',
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def charge_model(device,params='trained_param/qm9spectra/npacharge.pth'):
    '''Loads the QM9-spectra-trained per-atom NPA partial charge model.'''
    state_dict=torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=1,
                 irreps_out=None,
                 summation=False,
                 norm=False,
                 out_type='scalar',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def dipole_model(device,params='trained_param/qm9spectra/dipole.pth'):
    '''Loads the QM9-spectra-trained molecular dipole moment model.'''
    state_dict=torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=1,
                 irreps_out='1o',
                 summation=True,
                 norm=False,
                 out_type='dipole',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def polar_model(device,params='trained_param/qm9spectra/polar.pth'):
    '''Loads the QM9-spectra-trained static polarizability model.'''
    state_dict=torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='2e',
                 summation=True,
                 norm=False,
                 out_type='2_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def quadrupole_model(device,params='trained_param/qm9spectra/quadrupole.pth'):
    '''Loads the QM9-spectra-trained static quadrupole moment model.'''
    state_dict=torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='2e',
                 summation=True,
                 norm=False,
                 out_type='2_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def hyperpolar_model(device,params='trained_param/qm9spectra/hyperpolar.pth'):
    '''Loads the QM9-spectra-trained static (first) hyperpolarizability model.'''
    state_dict=torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='1o+3o',
                 summation=True,
                 norm=False,
                 out_type='3_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def octapole_model(device,params='trained_param/qm9spectra/octapole.pth'):
    '''Loads the QM9-spectra-trained static octapole moment model.'''
    state_dict=torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='1o+3o',
                 summation=True,
                 norm=False,
                 out_type='3_tensor',
                 grad_type=None,
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def Hi_model(device,params='trained_param/qm9spectra/Hi.pth'):
    '''Loads the QM9-spectra-trained atomic-block Hessian (grad_type='Hi') model.'''
    state_dict = torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=9,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='scalar',
                    grad_type='Hi',
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def Hij_model(device,params='trained_param/qm9spectra/Hij.pth'):
    '''Loads the QM9-spectra-trained interatomic-block Hessian (grad_type='Hij') model.'''
    state_dict = torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=9,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='scalar',
                    grad_type='Hij',
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def dedipole_model(device,params='trained_param/qm9spectra/dedipole.pth'):
    '''Loads the QM9-spectra-trained dipole-derivative (IR intensity precursor) model.'''
    state_dict=torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=1,
                 irreps_out='1o',
                 summation=False,
                 norm=False,
                 out_type='dipole',
                 grad_type='dipole',
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def depolar_model(device,params='trained_param/qm9spectra/depolar.pth'):
    '''Loads the QM9-spectra-trained polarizability-derivative (Raman intensity precursor) model.'''
    state_dict=torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                 act='swish',
                 maxl=3,
                 num_block=3,
                 radial_type='trainable_bessel',
                 num_radial=32,
                 attention_head=8,
                 rc=5.0,
                 dropout=0.0,
                 use_cutoff=False,
                 max_atomic_number=9,
                 atom_ref=None,
                 scale=1.0,
                 scalar_outsize=2,
                 irreps_out='2e',
                 summation=False,
                 norm=False,
                 out_type='2_tensor',
                 grad_type='polar',
                 device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def nmr_model(device,params):
    '''Loads a QM9-NMR-trained isotropic shielding model; `params` must be supplied explicitly (element-specific checkpoints, e.g. trained_param/qm9nmr/shield_iso_c.pth).'''
    state_dict = torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=9,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=1,
                    irreps_out=None,
                    summation=False,
                    norm=False,
                    out_type='scalar',
                    grad_type=None,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model

def uv_model(device,params='trained_param/qm9spectra/borden_os.pth'):
    '''Loads the QM9-spectra-trained UV-vis oscillator-strength model (240 lowest transitions).'''
    state_dict = torch.load(params, map_location=device)
    model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=9,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=240,
                    irreps_out=None,
                    summation=True,
                    norm=False,
                    out_type='scalar',
                    grad_type=None,
                    device=device)
    model.load_state_dict(state_dict=state_dict)
    return model




def qm9s_dynamic_polarizability_model(device,params='trained_param/dynamic-polarizability/QM9S_dynamic_polarizability.pth'):
    '''Loads the paper's QM9SPol dynamic polarizability model: 61 frequency points, UV-vis-conditioned (x_features=61). This is the model behind Table 1's QM9SPol (UV-vis input, 61 points) row.'''
    state_dict = torch.load(params, map_location=device)
    model = DetaNet(num_features=256,
                        act='swish',
                        maxl=3,
                        num_block=6, #3
                        radial_type='trainable_bessel',
                        num_radial=32,
                        attention_head=32,
                        rc=6,
                        dropout=0.0,
                        use_cutoff=False,
                        max_atomic_number=34,
                        atom_ref=None,
                        scale=1.0,
                        scalar_outsize=(4*61), # 2,#4, 
                        irreps_out= '122x2e', #'2e',# '2e+2e',
                        summation=True,
                        norm=False,
                        out_type='multi_tensor',
                        grad_type=None,
                        x_features=61,
                        device=device)
    model.load_state_dict(state_dict=state_dict)
    return model


def hopv241_dynamic_polarizability_model(device,params='trained_param/dynamic-polarizability/HOPV241_dynamic_polarizability.pth'):
    '''Loads the paper's best-performing HOPV15 dynamic polarizability model: 241 frequency points, UV-vis-conditioned (x_features=241). This is the model behind Table 1's HOPV15 (UV-vis input, 241 points) row -- see also Figure 5 in the paper.'''
    state_dict = torch.load(params, map_location=device)
    model = DetaNet(num_features=512,
                        act='swish',
                        maxl=3,
                        num_block=4, #3
                        radial_type='trainable_bessel',
                        num_radial=32,
                        attention_head=64,
                        rc=4,
                        dropout=0.0,
                        use_cutoff=False,
                        max_atomic_number=34,
                        atom_ref=None,
                        scale=1.0,
                        scalar_outsize=(4*241), # 2,#4, 
                        irreps_out= '482x2e', #'2e',# '2e+2e',
                        summation=True,
                        norm=False,
                        out_type='multi_tensor',
                        grad_type=None,
                        x_features=241,
                        device=device)
    model.load_state_dict(state_dict=state_dict)
    return model