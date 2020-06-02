mount:
	sshfs bayes:/home/niklas/repos/NIPS_2020_NMUX/data data

umount:
	fusermount -u data
