import re


def epoch_from_name(name):
    epoch, = re.findall('\d+', name)
    return int(epoch)


def latest_epoch(model_path):
    gen_checkpoints = model_path.glob("generator_*.h5")
    disc_checkpoints = model_path.glob("discriminator_*.h5")

    gen_epochs = [epoch_from_name(path.stem) for path in gen_checkpoints]
    disc_epochs = [epoch_from_name(path.stem) for path in disc_checkpoints]

    latest_gen_epoch = max(gen_epochs)
    latest_disc_epoch = max(disc_epochs)

    assert (
        latest_gen_epoch == latest_disc_epoch
    ), "Latest disc and gen epochs differ"

    return latest_gen_epoch


def load_weights(model, model_path, epoch=None):
    if epoch is None:
        epoch = latest_epoch(model_path)

    gen_checkpoint = model_path / f"generator_{epoch:05d}.h5"
    disc_checkpoint = model_path / f"discriminator_{epoch:05d}.h5"

    model.load_generator(gen_checkpoint)
    model.load_discriminator(disc_checkpoint)

    return epoch