"""

Copyright Luke Mondy, 2018

This file is covered by GNU General Public License v3.0
Please see: https://choosealicense.com/licenses/gpl-3.0/ for the full license

"""


import UWGeodynamics as GEO
import underworld as uw
import underworld.function as fn
from UWGeodynamics.surfaceProcesses import SedimentationThreshold
import numpy
import os.path
from mpi4py import MPI
import argparse

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_scale", action='store', default = 1.)
    parser.add_argument("--total_vel", action="store", required = True)
    #parser.add_argument("--time_to_stop", action="store", default = -1.)
    parser.add_argument("--gap_to_stop_sedi", action="store", required = True)
    parser.add_argument("--total_time", action="store", required = True)
    parser.add_argument("--name", action="store", required = True)

    return parser.parse_args()


u = GEO.UnitRegistry
GEO.rcParams["initial.nonlinear.tolerance"] = 1e-3
GEO.rcParams["nonlinear.tolerance"] =         1e-3
GEO.rcParams["nonlinear.min.iterations"] = 1
GEO.rcParams["CFL"] = 0.1
#GEO.rcParams["CFL"] = 0.5

GEO.rcParams["advection.diffusion.method"] = "SLCN"
GEO.rcParams["shear.heating"] = True
GEO.rcParams["surface.pressure.normalization"] = True  # Make sure the top of the model is approximately 0 Pa

GEO.rcParams["swarm.particles.per.cell.2D"] = 60
GEO.rcParams["popcontrol.split.threshold"] = 0.95
GEO.rcParams["popcontrol.max.splits"] = 100
GEO.rcParams["popcontrol.particles.per.cell.2D"] = 60

parser = vars(setup_args())

res_scale =        float(parser['res_scale'])
total_vel =        float(parser["total_vel"]) * u.cm / u.year
#time_to_stop =     float(parser["time_to_stop"]) * 1e6 * u.year
gap_to_stop_sedi = float(parser["gap_to_stop_sedi"]) * u.km
total_time =       float(parser["total_time"]) * 1e6 * u.year
name =                   parser['name']

default_resolution = (608,192)
resolution = tuple(map(lambda x: int(x * res_scale), default_resolution))

default_checkpoint = 200e3 * u.years
checkpoint_interval = default_checkpoint / total_vel.magnitude

output_dir = "lmr_res{}x{}_{}_totalvel-{}_gap_{}_{}".format(
        resolution[0], 
        resolution[1], 
        name,
        total_vel.magnitude,
        gap_to_stop_sedi.magnitude,
        total_time.magnitude)



# Characteristic values of the system
half_rate = 1. * u.centimeter / u.year
model_length = 600e3 * u.meter
model_height = 200e3 * u.meter
refViscosity = 1e21 * u.pascal * u.second
surfaceTemp = 273.15 * u.degK
baseModelTemp = 1603.15 * u.degK
bodyforce = 3200 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2
KT = (baseModelTemp - surfaceTemp)

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM
GEO.scaling_coefficients["[temperature]"] = KT


Model = GEO.Model(elementRes=resolution,
                  minCoord=(-300 * u.kilometer, -180 * u.kilometer),
                  maxCoord=( 300 * u.kilometer,  20 * u.kilometer),
                  gravity = (0., -9.81 * u.m / u.s**2),
                  outputDir = output_dir)



Model.diffusivity = 1e-6 * u.metre**2 / u.second 
Model.capacity    = 1000. * u.joule / (u.kelvin * u.kilogram)  # From:

air_shape         = GEO.shapes.Layer2D(top = Model.top,         bottom = 0.0)
uc_shape          = GEO.shapes.Layer2D(top = 0.0 * u.kilometer, bottom = -20*u.kilometer)
uc_markers_shape  = GEO.shapes.Layer2D(top = -5.*u.kilometer,   bottom = -10*u.kilometer)
#lc_shape         = GEO.shapes.MultiShape([GEO.shapes.Layer2D(top=-20*u.km, bottom=-40*u.kilometer), GEO.shapes.Box(minX=-5.*u.kilometer, maxX=5. * u.kilometer, top=-40*u.kilometer, bottom=-40*u.kilometer-10.*u.kilometer)])
lc_shape          = GEO.shapes.Layer2D(top=-20*u.kilometer,     bottom=-40*u.kilometer)
mantle_shape      = GEO.shapes.Layer2D(top=-40*u.kilometer,     bottom=-140*u.kilometer)
astheno_shape     = GEO.shapes.Layer2D(top=-140*u.kilometer,    bottom=Model.bottom)

crust_shape       =GEO.shapes.MultiShape([uc_shape, lc_shape]) 


air        = Model.add_material(name="air", shape=air_shape)
sediment   = Model.add_material(name="Sediment")
uc         = Model.add_material(name="Upper crust",  shape=uc_shape)
uc_markers =  Model.add_material(name="Upper crust Markers",  shape=uc_markers_shape )
mantle     = Model.add_material(name="Mantle",  shape=mantle_shape)
astheno    = Model.add_material(name="Asthenosphere",  shape=astheno_shape)
lc         =  Model.add_material(name="Lower crust",  shape=lc_shape)


air.diffusivity = 1.0e-5 * u.metre**2 / u.second
air.capacity = 100. * u.joule / (u.kelvin * u.kilogram)


air.density      =  0.1 * u.kilogram / u.metre**3
sediment.density = GEO.LinearDensity(reference_density=2700. * u.kilogram / u.metre**3)
uc.density       = GEO.LinearDensity(reference_density=2800. * u.kilogram / u.metre**3)
uc_markers.density       = uc.density
lc.density       = GEO.LinearDensity(reference_density=2900. * u.kilogram / u.metre**3)
mantle.density   = GEO.LinearDensity(reference_density=3370. * u.kilogram / u.metre**3)
astheno.density   = mantle.density


sediment.radiogenicHeatProd   = 1.2 * u.microwatt / u.meter**3
uc.radiogenicHeatProd         = 1.2 * u.microwatt / u.meter**3
lc.radiogenicHeatProd         = 0.6 * u.microwatt / u.meter**3
mantle.radiogenicHeatProd     = 0.02 * u.microwatt / u.meter**3
uc_markers.radiogenicHeatProd = uc.radiogenicHeatProd
astheno.radiogenicHeatProd = mantle.radiogenicHeatProd


rh = GEO.ViscousCreepRegistry()
air.viscosity         = 1e18 * u.pascal * u.second
sediment.viscosity    = rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
uc.viscosity          = rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990
lc.viscosity          = rh.Dry_Mafic_Granulite_Dislocation_Wang_et_al_2012
mantle.viscosity      = rh.Wet_Olivine_Dislocation_Hirth_and_Kohlstedt_2003
uc_markers.viscosity  = uc.viscosity
astheno.viscosity     = mantle.viscosity
 

pl = GEO.PlasticityRegistry()
uc.plasticity         = pl.Rey_and_Muller_2010_UpperCrust
uc.plasticity.frictionCoefficient = 0.55
uc.plasticity.frictionAfterSoftening = 0.055
uc_markers.plasticity = uc.plasticity
sediment.plasticity   = uc.plasticity
lc.plasticity         = pl.Rey_et_al_2014_LowerCrust
mantle.plasticity     = pl.Rey_et_al_2014_LithosphericMantle
astheno.plasticity    = mantle.plasticity

uc.minViscosity         = 1e19 * u.pascal * u.second
uc.maxViscosity         = 1e23 * u.pascal * u.second
sediment.minViscosity   = 1e19 * u.pascal * u.second
sediment.maxViscosity   = 1e23 * u.pascal * u.second
uc_markers.minViscosity = 1e19 * u.pascal * u.second
uc_markers.maxViscosity = 1e23 * u.pascal * u.second
lc.minViscosity         = 1e19 * u.pascal * u.second
lc.maxViscosity         = 1e23 * u.pascal * u.second
mantle.minViscosity     = 1e19 * u.pascal * u.second
mantle.maxViscosity     = 1e23 * u.pascal * u.second
astheno.minViscosity    = 1e19 * u.pascal * u.second
astheno.maxViscosity    = 1e23 * u.pascal * u.second

uc.stressLimiter       = 150 * u.megapascals
lc.stressLimiter       = 150 * u.megapascals
sediment.stressLimiter = 150 * u.megapascals
mantle.stressLimiter   = 300 * u.megapascals
uc_markers.stressLimiter = uc.stressLimiter
astheno.stressLimiter    = mantle.stressLimiter

"""
====================================
Partial melting

Note: notice the meltExpansion is set to 0. Since we don't have any melt transport (all melt is held in place)
      we wouldn't then expect any density change.
"""

# Solidus
solidii = GEO.SolidusRegistry()
crust_solidus = solidii.Crustal_Solidus
mantle_solidus = solidii.Mantle_Solidus

#Liquidus
liquidii = GEO.LiquidusRegistry()
crust_liquidus = liquidii.Crustal_Liquidus
mantle_liquidus = liquidii.Mantle_Liquidus


sediment.add_melt_modifier(crust_solidus, crust_liquidus, 
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.0, 
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = -1.0e3
                        )
uc.add_melt_modifier(crust_solidus, crust_liquidus, 
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.0, 
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = -1.0e3
                        )

uc_markers.add_melt_modifier(crust_solidus, crust_liquidus, 
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.0, 
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = -1.0e3
                        )

lc.add_melt_modifier(crust_solidus, crust_liquidus, 
                         latentHeatFusion=250.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.3,
                         meltExpansion=0.0, 
                         viscosityChangeX1 = 0.15,
                         viscosityChangeX2 = 0.30,
                         viscosityChange = -1.0e3
                        ) 

mantle.add_melt_modifier(mantle_solidus, mantle_liquidus, 
                         latentHeatFusion=450.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.02,
                         meltExpansion=0.0, 
                         viscosityChangeX1 = 0.00,
                         viscosityChangeX2 = 0.02,
                         viscosityChange = -1.0e2
                        ) 

astheno.add_melt_modifier(mantle_solidus, mantle_liquidus, 
                         latentHeatFusion=450.0 * u.kilojoules / u.kilogram / u.kelvin,
                         meltFraction=0.,
                         meltFractionLimit=0.02,
                         meltExpansion=0.0, 
                         viscosityChangeX1 = 0.00,
                         viscosityChangeX2 = 0.02,
                         viscosityChange = -1.0e2
                        ) 
"""
Partial melting
====================================
"""


"""
====================================
Passive tracers
"""

# Put a single passive tracer per element. The model will check if they get 
# spaced out (see the post_hook function).
x = numpy.linspace(Model.minCoord[0], Model.maxCoord[0], resolution[0]) * u.kilometer
y = -40. * u.kilometer

moho_tracers = Model.add_passive_tracers(name="Moho", vertices=[x,y])


x_c, y_c = GEO.circles_grid(radius=2.0*u.kilometer, 
                    minCoord=[Model.minCoord[0], lc.bottom], 
                    maxCoord=[Model.maxCoord[0], 0.*u.kilometer])
circles_c = Model.add_passive_tracers(name="FSE_Crust", vertices=[x_c, y_c])


x_m, y_m = GEO.circles_grid(radius=2.0*u.kilometer, 
                    minCoord=[Model.minCoord[0], Model.bottom], 
                    maxCoord=[Model.maxCoord[0], mantle.top])
circles_m = Model.add_passive_tracers(name="FSE_Mantle", vertices=[x_m, y_m])
"""
Passive tracers
====================================
"""


"""
====================================
Initial conditions
"""

"""
====================================
  Random damage
"""
def gaussian(xx, centre, width):
    return ( numpy.exp( -(xx - centre)**2 / width ))

maxDamage = 0.2
centre = (GEO.nd(0. * u.kilometer), GEO.nd(-20. * u.kilometer))
width = GEO.nd(5. * u.kilometer)  # this gives a normal distribution
                                  # from about -100 km to 100 km

Model.plasticStrain.data[:] = maxDamage * numpy.random.rand(*Model.plasticStrain.data.shape[:])
Model.plasticStrain.data[:,0] *= gaussian(Model.swarm.particleCoordinates.data[:,0], centre[0], width)
Model.plasticStrain.data[:,0] *= gaussian(Model.swarm.particleCoordinates.data[:,1], centre[1], width*100)

# The following lines make the random damage only apply to the crust
noncrust_mask = Model.swarm.particleCoordinates.data[:,1] <= GEO.nd(-40 * u.kilometer)
air_mask = Model.swarm.particleCoordinates.data[:,1] > GEO.nd(0 * u.kilometer)

Model.plasticStrain.data[noncrust_mask] = 0.0
Model.plasticStrain.data[air_mask] = 0.0
"""
  Random damage
====================================
"""

# Temp initial conditions
Model.set_temperatureBCs(top=293.15 * u.degK, 
                         bottom=1623.15 * u.degK, 
                         nodeSets = [(air_shape, 293.15 * u.degK), (astheno_shape, 1623.15 * u.degK)])

# We need to initialise the model first for two reasons:
# 1: we need to calculate the steady-state geotherm, based on the above initial conditions
# 2: so that the lithostatic pressure calculation is correct, since density is temperature dependent
Model.init_model()
"""
Initial conditions
====================================
"""

"""
====================================
Boundary conditions
"""
# Reset the temp boundary conditions to be something we want during the geodynamics
Model.set_temperatureBCs(top=293.15 * u.degK, 
                         bottom=1623.15 * u.degK)


# Making the air compressible means that its volume can change.
# This is good, because it allows us to set a freeslip BC on the top of the model,
# which is good because it gives the model a 'reference' point. You can imagine that
# if both vertical walls are free slip, the top is 'open', and the bottom has a pressure
# BC, it can be hard for the model to know where it exists (given that stokes only uses
# the pressure gradient). By fixing the top, it can figure this out.

air.compressibility = 1e4  # Not sure what's a good value is

P, bottomPress = Model.get_lithostatic_pressureField()


# Get the average of the pressure along the bottom, and make it a pressure BC along the bottom
bottomPress = GEO.Dimensionalize(numpy.average(bottomPress), u.megapascal).magnitude
print("Initially calculated bottom pressure:", bottomPress)

"""
# Restarting models with pressure boundary conditions can be a bit dangerous. To avoid this,
# we write out the bottom pressure to a file. When the model restarts, we check the file, and 
# use the pressure it has, instead of calculating our own.
pressureBC_file = "./pressureBC_{}.dat".format(output_dir)
if uw.rank() == 0:
    if os.path.isfile(pressureBC_file):
        # If a file with the bottom pressure already exists, read from it.
        with open(pressureBC_file, 'r') as f:
            bottomPress = numpy.float64(f.readline().strip())
        print("Loaded bottom pressure BC from file: ", bottomPress)
    else:
        # If no existing pressure is around, write the pressure we calculated to the file
        with open(pressureBC_file, 'w') as f:
            f.write("{:.12f}".format(bottomPress))
        print("Saved bottom pressure BC to file")
else:
    bottomPress = None

# since only 1 CPU got the file, send it out to all CPUs
bottomPress = MPI.COMM_WORLD.bcast(bottomPress, root=0)
uw.barrier()  # wait for them to catchup
"""
# until restarts with basal pressure BCs is fixed, removing all this
bottomPress = bottomPress * u.megapascal  # then make it a unit


Model.set_velocityBCs(
                      left  = [total_vel * -0.5, 0. * u.centimetre / u.year], 
                      right = [total_vel * 0.5,  0. * u.centimetre / u.year], 
                      top   = [None,             0. * u.centimetre / u.year],
                      )

Model.set_stressBCs(
        bottom = [0., bottomPress],
        )


def post_hook():
    """
    Stop any brittle yielding near the edges of the model
    """
    coords = fn.input()
    zz = (coords[0] - GEO.nd(Model.minCoord[0])) / (GEO.nd(Model.maxCoord[0]) - GEO.nd(Model.minCoord[0]))
    fact = fn.math.pow(fn.math.tanh(zz*20.0) + fn.math.tanh((1.0-zz)*20.0) - fn.math.tanh(20.0), 4)
    Model.plasticStrain.data[:] = Model.plasticStrain.data[:] * fact.evaluate(Model.swarm)

    """
    Check spacing for when sedimentation should turn off
    # This solution was provided by: https://stackoverflow.com/a/38008452
    """
    rank = uw.rank()
    root = 0

    # get all the moho tracers that are on our CPU, in x sorted order
    local_array = numpy.sort(moho_tracers.swarm.particleCoordinates.data[:,0])
    sendbuf = numpy.array(local_array)

    # We have to figure out how many particles each CPU has, and let the root
    # cpu know
    sendcounts = numpy.array(MPI.COMM_WORLD.gather(len(sendbuf), root))

    if rank == root:
        # prepare to receive all this data
        recvbuf = numpy.empty(sum(sendcounts), dtype=float)
    else:
        recvbuf = None

    # Gather up all the data and put it in recvbuf
    MPI.COMM_WORLD.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)
    if rank == root:
        # find the biggest gap in the X direction in the moho_tracers
        diff = numpy.max(numpy.diff(numpy.sort(recvbuf)))  # recvbuf is the array of all particles
    else:
        diff = None

    # Now that we know the biggest gap, tell all the other CPUs
    diff = MPI.COMM_WORLD.bcast(diff, root=0)
    biggest_gap = GEO.Dimensionalize(diff, u.km)
    uw.barrier()
    print(uw.rank(), "Biggest gap in tracers", biggest_gap)

    if biggest_gap > gap_to_stop_sedi:
        print("Sedimentation turned: OFF")
        threshold = -10 * u.kilometers
    else:
        print("Sedimentation turned: ON")
        threshold = -1 * u.kilometers

    Model.surfaceProcesses = GEO.surfaceProcesses.SedimentationThreshold(air=[air], sediment=[sediment], threshold=threshold)


Model.postSolveHook = post_hook


"""
Boundary conditions
====================================
"""

GEO.rcParams["default.outputs"].append("projMeltField")
GEO.rcParams["default.outputs"].append("projStressTensor")


# This is a bunch of solver options. You can try playing with them, but these should be good enough.
solver = Model.solver

# Decide whether to use mumps or multigrid
if resolution[0] * resolution[1] < 1e6:
    print("Using mumps")
    solver.set_inner_method("mumps")
else:
    print("Using multigrid with coarse mumps")
    #solver.options.mg.levels = 4
    solver.options.A11.mg_coarse_pc_factor_mat_solver_package = "mumps"
    solver.options.A11.mg_coarse_pc_type = "lu"
    solver.options.A11.mg_coarse_ksp_type = "preonly"
    #solver.options.A11.mg_coarse_ksp_view = ""

solver.options.A11.ksp_rtol=1e-8
solver.options.A11.ksp_set_min_it_converge = 10
solver.options.A11.use_previous_guess = True
solver.options.scr.ksp_rtol=1e-6
solver.options.scr.use_previous_guess = True
solver.options.scr.ksp_set_min_it_converge = 10
solver.options.scr.ksp_type = "cg"

#solver.options.main.help = ""
solver.options.main.remove_constant_pressure_null_space=True
#solver.options.main.Q22_pc_type = "uwscale"
solver.set_penalty(0)



#Model.solve()
#Model.run_for(10.0e6* u.year, dt=10e3 * u.year, checkpoint_interval=100e3*u.years)
#Model.run_for(total_time, restartStep=-1, checkpoint_interval=checkpoint_interval)
Model.run_for(total_time, checkpoint_interval=checkpoint_interval)



