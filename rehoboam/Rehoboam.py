from keras.models import Sequential
from keras.optimizers import Adam
from rehoboam.discriminator import build_discriminator
from rehoboam.generator import build_generator

adam = Adam(lr=0.0002)

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

generator = build_generator()

Rehoboam = Sequential(name="Rehoboam")
discriminator.trainable = False
Rehoboam.add(generator)
Rehoboam.add(discriminator)

Rehoboam.compile(loss='binary_crossentropy', optimizer=adam)

Rehoboam.summary()
