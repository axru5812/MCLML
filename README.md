COMPUTING THE SPECTRUM:

compute fesc/bin (bin at 20 kms /1000photons per bin) of col 1

gen a continuum + line gaussian intrinsic

weight that by fesc.


for an input velocity
	calculate fesc
	histogram the photons from col 2
	multiply by escape fraction
	multiply by the relative flux (intrinsic line)

