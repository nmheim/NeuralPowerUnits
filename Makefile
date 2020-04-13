mount:
	sshfs bayes:/home/niklas/repos/NIPS_2020_NMUX/data data
	sshfs bayes:/home/niklas/repos/NIPS_2020_NMUX/plots plots

umount:
	fusermount -u data
	fusermount -u plots
