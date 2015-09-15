!
! Illustrate a Fortran 2003 style C-Fortran interface
! It has been more than a decade since the standard --
! most compilers should support it at this point.
!
subroutine do_block(lda, A, B, C, i, j, k, BLOCK_SIZE)
    implicit none

    integer, intent(in) :: lda, i, j, k, BLOCK_SIZE
    real*8, intent (inout) :: A(0:lda-1, 0:lda-1)
    real*8, intent (inout) :: B(0:lda-1, 0:lda-1)
    real*8, intent (inout) :: C(0:lda-1, 0:lda-1)

    integer :: M, N, Kbig, MM, NN, KK

    M = i+merge(lda-i, BLOCK_SIZE, (i+BLOCK_SIZE) .gt. lda)
    N = j+merge(lda-j, BLOCK_SIZE, (j+BLOCK_SIZE) .gt. lda)
    Kbig = k+merge(lda-k, BLOCK_SIZE, (k+BLOCK_SIZE) .gt. lda)

    MM = merge(mod(M, BLOCK_SIZE), BLOCK_SIZE, mod(M, BLOCK_SIZE) .gt. 0)
    NN = merge(mod(N, BLOCK_SIZE), BLOCK_SIZE, mod(N, BLOCK_SIZE) .gt. 0)
    KK = merge(mod(Kbig, BLOCK_SIZE), BLOCK_SIZE, mod(Kbig, BLOCK_SIZE) .gt. 0)

    call dgemm(A( i:M-1, k:Kbig-1), MM, KK, B(k:Kbig-1, j:N-1), KK, NN, C(i:M-1, j:N-1), MM, NN)

end subroutine do_block

subroutine square_dgemm(M, A, B, C) bind(C)

    use, intrinsic :: iso_c_binding
    implicit none

    integer (c_int), value :: M
    integer, parameter :: BLOCK_SIZE = 512
    real (c_double), intent (inout), dimension(*) :: A
    real (c_double), intent (inout), dimension(*) :: B
    real (c_double), intent (inout), dimension(*) :: C

    integer :: n_blocks

    integer :: i, j, k, bi, bj, bk

    n_blocks = M / BLOCK_SIZE + merge(1, 0, mod(M,BLOCK_SIZE).gt.0)

    do bj = 0, n_blocks-1
        j = bj * BLOCK_SIZE
        do bk = 0, n_blocks-1
            k = bk * BLOCK_SIZE
            do bi = 0, n_blocks-1
                i = bi * BLOCK_SIZE
!                write(*,*) 'got here'
                call do_block(M, A, B, C, i, j, k, BLOCK_SIZE)
!                write(*,*) 'got here'
            end do
        end do
    end do

end subroutine square_dgemm
