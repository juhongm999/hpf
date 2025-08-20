" #################################################################
" ## Commands used in this ~/.vimrc - START                      ##
" #################################################################
"
" :echo expand("%")     :get current filename
" :echo expand("*")     :get filenames in current directory
"
" ctrl+r+"              :paste in insert mode
"
" :find <filename>      :open file (saved to buffer)
" :e <filename>         :open file from current dir (saved to buffer)
" :ls                   :see buffered file list
" :b <filename>|<no>    :open file (from buffer)
" :bn                   :open next buffer
" :bp                   :open previous buffer
" :bd                   :delete current buffer
" :bd <filename>|<no>   :delete current buffer
" :bd!                  :delete current buffer without saving changes
"
" ctrl+w+ v (:vs fname) :vim vertical split (copy)
" ctrl+w+ s (:sp fname) :vim horizontal split (copy)
" ctrl+w+ {hjkl}        :vim move btw windows
" ctrl+w+ {-+}          :vim vertical window adjust
" ctrl+w+ {<>}          :vim horizontal window adjust
" ctrl+w+ =             :vim reset window size
"
""" VIM command basic (must memorize)...
" U                     :return the 'line' to its original state
" Y                     :copy a line (=yy)
" H                     :move cursor to top
" M                     :move cursor to middle
" L                     :move cursor to bottom
" C                     :change whole line (=cc)
" R                     :replace mode (insert in windows)
" %                     :find matching parenthesis
" <number>G             :goto line number (=:<number>)
" <C-a>                 :increment digit
" <C-g>                 :show file location/status
" <C-o>                 :go to older position (using after searching)
" <C-i>                 :go to newer position
" m<char/digit>         :mark current position
" `<char/digit>         :goto mark
" q<char/digit>         :record macro (exit with q)
" <digit>@<char/digit>  :run macro <digit> times
" :reg                  :view clipboard (registers)
" :%s/old/new/g         :replace old with new
" :%s/old/new/gc        :replace old with new (ask every time)
" :s/old/new/g          :replace old with new in a line
" :!<external cmd>      :run terminal command
" :r <file>             :retrieve file and append contents below cursor
" :r !ls                :retrieve file list and append below cursor
" <v select>:w <file>   :write visual-selected contents to file
" :'<,'>!sort           :sort visual-selected lines
" .,.+4!sort            :sort 4 lines below cursor
" :'<,'>!column -t      :organize visual-selected lines in table form
"
""" Tab commands (bind later)
" :tabnew               :add new tab
" :tabclose             :closetab
" :<digit>gt            :go to <digit>-th tab
" :tabfirst             :go to first tab
" :tablast              :go to last tab
" :tabs                 :view open tabs
"
""" With c, d, y, you can append... (i-in, a-all)
" iw                    :inner word
" it                    :inner tag (for HTML)
" i"                    :inner quotes
" a"                    :everything including quotes
" i)                    :inner parenthesis
" a)                    :everything including parenthesis
" ip                    :inner paragraph
" as                    :a sentence
" /sentence             :upto sentence (upto . or space)
"    
" #################################################################
" ## Commands used in this ~/.vimrc - END                        ##
" #################################################################







" #################################################################
" ## Plugin configuration - START                                ##
" #################################################################

" #########################
" ## Vundle plugins      ##
" #########################
" :PluginInstall : install plugins
" :PluginList    : see all installed plugins
" :PluginUpdate  : update (delete) plugins
" below is required by Vundle to operate properly
set nocompatible
filetype off
set rtp+=~/.vim/bundle/Vundle.vim

call vundle#begin()
    Plugin 'VundleVim/Vundle.vim'           " let Vundle manage Vundle, required
    Plugin 'flazz/vim-colorschemes'         " vim color schemes
    Plugin 'vim-airline/vim-airline'        " visualize vim modes, tabs, etc..
    Plugin 'vim-airline/vim-airline-themes' " (required for vim-airline)
    Plugin 'szw/vim-maximizer'              " maximizing vim window
    Plugin 'scrooloose/nerdtree'            " directory navigator
    Plugin 'tpope/vim-surround'             " easy parenthesis
call vundle#end()
filetype plugin indent on


" #########################
" ## Vimspector          ##
" #########################
fun FocusCode(id)
    call win_gotoid(a:id.code)
endfun

fun FocusOutput(id)
    call win_gotoid(a:id.output)
endfun

fun SetVimspectorUI(id)
    vertical resize 150
    call win_gotoid(a:id.output)
    resize 20
    call win_gotoid(a:id.code)
    "call win_gotoid(a:id.terminal)
endfun


function! s:CustomiseUI()
    call win_gotoid(g:vimspector_session_windows.variables)
    q
    call win_gotoid(g:vimspector_session_windows.watches)
    q
    call win_gotoid(g:vimspector_session_windows.stack_trace)
    q
    call SetVimspectorUI(g:vimspector_session_windows)
    "NERDTreeToggle

    "call win_gotoid(g:vimspector_session_windows.code)
endfunction

augroup MyVimspectorUICustomistaion
    autocmd!
    autocmd User VimspectorUICreated call s:CustomiseUI()
augroup END

let g:vimspector_enable_mappings = 'HUMAN'
packadd! vimspector
nmap <leader>dd  :call vimspector#Launch()<CR>
nmap <leader>dx  :call vimspector#Reset()<CR>
nmap <leader>dr  :call vimspector#Restart()<CR>
nmap <leader>dp  :call vimspector#Pause()<CR>
nmap <leader>dcb :call vimspector#ClearBreakpoints()<CR>
nmap <leader>du  :call vimspector#RunToCursor()<CR>
nmap <leader>dp  :call vimspector#Pause()<CR>
nmap <leader>di <Plug>VimspectorBalloonEval
xmap <leader>di <Plug>VimspectorBalloonEval

nmap <leader>b   :call vimspector#ToggleBreakpoint()<CR>
nmap <leader>0   :call vimspector#StepOver()<CR>
nmap <leader>-   :call vimspector#StepInto()<CR>
nmap <leader>=   :call vimspector#StepOut()<CR>
nmap <leader>5   :call vimspector#Continue()<CR>

nmap <leader>dfu  :call vimspector#UpFrame()<CR>
nmap <leader>dfd  :call vimspector#DownFrame()<CR>

nmap <leader>q   :call FocusCode(g:vimspector_session_windows)<CR>
nmap <leader>a   :call FocusOutput(g:vimspector_session_windows)<CR>i

nmap <leader>]   :call SetVimspectorUI(g:vimspector_session_windows)<CR>

nmap <leader>fr  ^<C-v>$y<leader>a<C-r>"<CR><leader>q
vmap <leader>fc  y<esc><leader>a<C-r>"
vmap <leader>fv  y<esc><leader>a<C-r>"<CR><leader>q
nmap <leader>fc  yiw<leader>a<C-r>"
nmap <leader>fv  yiw<leader>a<C-r>"<CR><leader>q
vmap <leader>fs  y<esc><leader>a<C-r>".size()<CR><leader>q
nmap <leader>fs  yiw<leader>a<C-r>".size()<CR><leader>q
nmap <leader>fl  yiw<leader>alen(<C-r>")<CR><leader>q
vmap <leader>fl  y<leader>alen(<C-r>")<CR><leader>q

" #########################
" ## Colorscheme (favs.) ##
" #########################
"colorscheme monokai-chris
"colorscheme Monokai
"colorscheme candycode
"colorscheme codedark
colorscheme desertink

" #########################
" ## Maximizer           ##
" #########################
nnoremap <leader>m :MaximizerToggle<CR>

" #########################
" ## NERDTree            ##
" #########################
" m - opens NERDTree menu (can delete, edit filenames)
" t - open in a new tab
nmap <leader>ww  :NERDTreeToggle<CR>
nmap <leader>wf  :NERDTreeFocus<CR>
nmap <leader>wc  :NERDTreeCWD<CR>

" #########################
" ## Vim-Surround        ##
" #########################
" cs"'   - change surrounding from " to '
" cs'<q> - change surrounding from ' to <q></q>
" cst"   - change surrounding from tag to "
" ds"    - delete surrounding "
" ysiw"  - add surrounding " (inner word)
" yss"   - add surrounding " (line)
" S"     - add surrounding " in visual mode

" #################################################################
" ## Plugin configuration - END                                  ##
" #################################################################





" #################################################################
" ## Custom .vimrc settings - START                              ##
" #################################################################

" #########################
" ##  1. Visualization   ##
" #########################
syntax on
set number              "show line number
set relativenumber      "show relative line number
set incsearch           "highlight the first occurrence when searching
set hlsearch            "highlight all the search results
set cursorline          "show current cursorline
set nowrap              "no wrapping to next line
set tabstop=4           "tab size (only for rendering, has no effect on actual text)
set showmatch           "show matching brackets
set scrolloff=7         "show some lines after/before EOF
let python_highlight_all = 1

hi LineNrAbove guifg=red ctermfg=red
hi LineNrBelow guifg=green ctermfg=green

set nofoldenable        "show no folds when opening files
set foldmethod=indent   "fold codeblock based on indentation
set foldnestmax=20      "maximum number of nested folds
set foldlevel=15        "leave 15 parents blocks unfolded

set ignorecase
set smartcase
" /copyright      " Case insensitive
" /Copyright      " Case sensitive
" /copyright\C    " Case sensitive
" /Copyright\c    " Case insensitive

" <spacebar>: fold and unfold code blocks
nmap <space> za
nmap <leader><space> :nohlsearch<CR>

" highlights background when using multiple windows
augroup BgHighlight
    autocmd!
    autocmd WinEnter * set colorcolumn=80
    autocmd WinLeave * set colorcolumn=0
augroup END

" #########################
" ##  2. Text edit       ##
" #########################
set softtabstop=4       "tab size effective while editing
set shiftwidth=4        "tab size (<< and >>)
set expandtab           "tab becomes the number of spaces
filetype indent on
set autoindent          "automatic indent
set backspace=indent,eol,start

" python commenting ([c]omment, [a]dd, [d]elete, [b]lock, [h]eader)
nmap <leader>ca     ^i#<space><esc>
nmap <leader>cd     ^dw
vmap <leader>ca     ^<C-v>I#<space><esc>
vmap <leader>cd     ^<C-v>f<space>x
nmap <leader>cba    ^i##<space><esc>A<space>##<esc>Ypv$r#YkPj^w
nmap <leader>cbd    ^w3X$F<space>Djddkkdd
nmap <leader>cha    ^ir"""<space><esc>A<space>"""<esc>^
nmap <leader>chd    dW$F<space>D

" #########################
" ##  3. Files           ##
" #########################
set noswapfile          "no .swp files created during editing

" #########################
" ##  4. Commands        ##
" #########################
set showcmd             "show commands at the bottom
set wildmenu            "wild menu : with tab pressed
set path+=**            "search through every subdirectory

" #########################
" ##  5. Terminal        ##
" #########################
" <C-w>N                :go to normal mode in terminal
" i                     :go to insert mode in terminal
nmap <leader>\vt     :below vertical terminal<CR>
nmap <leader>\st     :below terminal<CR>

" #########################
" ##  6. sourcing vimrc  ##
" #########################
nmap <leader>\\v    :source ~/.vimrc<CR>

" #################################################################
" ## Custom .vimrc settings - END                                ##
" #################################################################
