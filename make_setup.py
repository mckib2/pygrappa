from textwrap import dedent


if __name__ == "__main__":
    tmpl = dedent(f'''\
    """Python GRAPPA image reconstruction."""
    
    from setuptools import setup
        
    setup(
        author='Nicholas McKibben',
        author_email='nicholas.bgp@gmail.com',
        url='https://github.com/mckib2/pygrappa',
        license='GPLv3',
        version='0.26.0',
        description='GeneRalized Autocalibrating Partially Parallel Acquisitions.',
        long_description=open('README.rst', encoding='utf-8').read(),
        packages=['pygrappa'],
        keywords=[
            'mri grappa parallel-imaging image-reconstruction python '
            'tgrappa slice-grappa sms split-slice-grappa vc-grappa '
            'igrappa hp-grappa segmented-grappa grappa-operator '
            'through-time-grappa pars grog nonlinear-grappa g-factor'
            'sense', 'cg-sense'],
    )
    ''')
    print(tmpl)
